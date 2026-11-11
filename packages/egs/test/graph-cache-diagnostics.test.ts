import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    VOLATILE_GRAPH_CACHE_LIMIT,
    buildRenderGraphKey,
    diagnoseFrameSize,
    diagnoseRenderGraphCache,
    type FrameSizeSnapshot,
    type GraphCachePluginSnapshot,
    type RenderGraphSnapshot,
} from '../src/fx/GraphCacheDiagnostics.ts';

/**
 * Independent re-implementation of the key formula written straight from
 * PostPipeline.createRenderGraph / HashKeyBuilder source:
 *
 *   hasher.raw(backend).bool(driven).bool(occlusion).bool(msaa)
 *   then per plugin: hasher.raw(PLUGIN_NAME) + updateGraphHash fragment
 *
 * bool(x) appends '1' / '0'; raw(x) appends String(x).
 */
function expectedKey(snapshot: RenderGraphSnapshot): string {
    const effectiveMSAA = snapshot.msaa && snapshot.backend !== 'WEBGL_JS';
    let key = `${snapshot.backend}${snapshot.gpuDrivenEnabled ? '1' : '0'}${
        snapshot.occlusionCullingEnabled ? '1' : '0'
    }${effectiveMSAA ? '1' : '0'}`;
    for (const plugin of snapshot.plugins) {
        key += `${plugin.name}${plugin.hashFragment}`;
    }
    return key;
}

function baseSnapshot(overrides: Partial<RenderGraphSnapshot> = {}): RenderGraphSnapshot {
    return {
        backend: 'WEBGL2_JS',
        gpuDrivenEnabled: false,
        occlusionCullingEnabled: false,
        msaa: false,
        plugins: [],
        stableCacheKeys: [],
        volatileCacheKeys: [],
        hasAnyShadow: false,
        ...overrides,
    };
}

const plugins: GraphCachePluginSnapshot[] = [
    // fragments mirror real updateGraphHash shapes: bool runs, raw numbers, no-ops
    { name: 'shadowMap', hashFragment: '71' }, // raw(shadowHashKey=7) + bool(useProxy=true)
    { name: 'forward', hashFragment: '10010' }, // five bool(...) calls
    { name: 'highlight', hashFragment: '3' }, // raw(groups.length = 3)
    { name: 'AO', hashFragment: '' }, // no-op updateGraphHash
];

describe('diagnoseRenderGraphCache - normal path', () => {
    it('builds a key identical to the source formula, bit for bit', () => {
        const snapshot = baseSnapshot({
            backend: 'WEBGPU_WASM',
            gpuDrivenEnabled: true,
            occlusionCullingEnabled: true,
            msaa: true,
            plugins,
        });
        assert.equal(buildRenderGraphKey(snapshot), expectedKey(snapshot));
        assert.equal(
            buildRenderGraphKey(snapshot),
            'WEBGPU_WASM111' + 'shadowMap71forward10010highlight3AO',
        );
    });

    it('reports a miss and a stable write when neither cache holds the key', () => {
        const snapshot = baseSnapshot({ plugins, hasAnyShadow: false });
        const result = diagnoseRenderGraphCache(snapshot);

        assert.equal(result.graphKey, expectedKey(snapshot));
        assert.equal(result.cacheHit, false);
        assert.equal(result.hitCacheKind, null);
        assert.equal(result.writeCacheKind, 'stable');
        assert.equal(result.volatileClearTriggered, null);
        assert.equal(result.msaaInvalidated, false);
        assert.equal(result.effectiveMSAA, false);
    });

    it('reports a miss and a volatile write when the scene has shadows', () => {
        const snapshot = baseSnapshot({ plugins, hasAnyShadow: true });
        const result = diagnoseRenderGraphCache(snapshot);

        assert.equal(result.cacheHit, false);
        assert.equal(result.writeCacheKind, 'volatile');
        assert.equal(result.volatileClearTriggered, false);
        assert.equal(result.volatileCacheSizeBefore, 0);
        assert.equal(result.volatileCacheSizeAfter, 1);
    });

    it('detects a stable cache hit and performs no write', () => {
        const snapshot = baseSnapshot({ plugins });
        const key = expectedKey(snapshot);
        snapshot.stableCacheKeys = ['some-other-key', key];

        const result = diagnoseRenderGraphCache(snapshot);
        assert.equal(result.graphKey, key);
        assert.equal(result.cacheHit, true);
        assert.equal(result.hitCacheKind, 'stable');
        assert.equal(result.writeCacheKind, null);
        assert.equal(result.volatileClearTriggered, null);
        assert.equal(result.volatileCacheSizeAfter, null);
    });

    it('detects a volatile cache hit and performs no write', () => {
        const snapshot = baseSnapshot({ plugins, hasAnyShadow: true });
        const key = expectedKey(snapshot);
        snapshot.volatileCacheKeys = [key];

        const result = diagnoseRenderGraphCache(snapshot);
        assert.equal(result.cacheHit, true);
        assert.equal(result.hitCacheKind, 'volatile');
        assert.equal(result.writeCacheKind, null);
    });

    it('prefers the stable cache when a key exists in both tables', () => {
        const snapshot = baseSnapshot({ plugins });
        const key = expectedKey(snapshot);
        snapshot.stableCacheKeys = [key];
        snapshot.volatileCacheKeys = [key];

        const result = diagnoseRenderGraphCache(snapshot);
        assert.equal(result.cacheHit, true);
        assert.equal(result.hitCacheKind, 'stable');
    });

    it('changes the key whenever any boolean input or plugin changes', () => {
        const baseline = baseSnapshot({ plugins, msaa: true });
        const baselineKey = diagnoseRenderGraphCache(baseline).graphKey;

        const toggles: Partial<RenderGraphSnapshot>[] = [
            { gpuDrivenEnabled: true },
            { occlusionCullingEnabled: true },
            { msaa: false },
            { backend: 'WEBGL2_WASM' },
        ];
        for (const toggle of toggles) {
            const other = baseSnapshot({ plugins, msaa: true, ...toggle });
            assert.notEqual(diagnoseRenderGraphCache(other).graphKey, baselineKey);
        }

        const fewerPlugins = baseSnapshot({ msaa: true, plugins: plugins.slice(0, 2) });
        assert.notEqual(diagnoseRenderGraphCache(fewerPlugins).graphKey, baselineKey);
    });

    it('returns plain JSON-serializable data without engine objects', () => {
        const result = diagnoseRenderGraphCache(baseSnapshot({ plugins, hasAnyShadow: true }));
        const roundTripped = JSON.parse(JSON.stringify(result));
        assert.deepEqual(roundTripped, result);
    });
});

describe('diagnoseRenderGraphCache - negative path (WEBGL_JS MSAA)', () => {
    it('invalidates requested MSAA on WEBGL_JS and reflects it in the key', () => {
        const withMsaa = baseSnapshot({ backend: 'WEBGL_JS', msaa: true, plugins });
        const result = diagnoseRenderGraphCache(withMsaa);

        assert.equal(result.requestedMSAA, true);
        assert.equal(result.effectiveMSAA, false);
        assert.equal(result.msaaInvalidated, true);
        // key must carry the effective (disabled) MSAA bit '0'
        assert.equal(result.graphKey, expectedKey(withMsaa));
        assert.match(result.graphKey, /^WEBGL_JS000/);
    });

    it('produces the same key with MSAA on or off on WEBGL_JS', () => {
        const on = diagnoseRenderGraphCache(baseSnapshot({ backend: 'WEBGL_JS', msaa: true, plugins }));
        const off = diagnoseRenderGraphCache(baseSnapshot({ backend: 'WEBGL_JS', msaa: false, plugins }));
        assert.equal(on.graphKey, off.graphKey);
        assert.equal(off.msaaInvalidated, false);
    });

    it('keeps MSAA effective on non-WEBGL_JS backends', () => {
        const result = diagnoseRenderGraphCache(
            baseSnapshot({ backend: 'WEBGL2_JS', msaa: true, plugins }),
        );
        assert.equal(result.effectiveMSAA, true);
        assert.equal(result.msaaInvalidated, false);
        assert.match(result.graphKey, /^WEBGL2_JS001/);
    });
});

describe('diagnoseRenderGraphCache - volatile clear boundary', () => {
    function shadowMiss(volatileCount: number) {
        const snapshot = baseSnapshot({
            plugins,
            hasAnyShadow: true,
            volatileCacheKeys: Array.from({ length: volatileCount }, (_, i) => `old-key-${i}`),
        });
        return { snapshot, result: diagnoseRenderGraphCache(snapshot) };
    }

    it('does NOT clear with 19 existing entries (20 after write is the limit)', () => {
        const { result } = shadowMiss(19);
        assert.equal(result.cacheHit, false);
        assert.equal(result.writeCacheKind, 'volatile');
        assert.equal(result.volatileCacheSizeBefore, 19);
        assert.equal(result.volatileClearTriggered, false);
        assert.equal(result.volatileCacheSizeAfter, VOLATILE_GRAPH_CACHE_LIMIT);
    });

    it('clears the whole table with 20 existing entries (21 after write)', () => {
        const { result } = shadowMiss(VOLATILE_GRAPH_CACHE_LIMIT);
        assert.equal(result.volatileCacheSizeBefore, 20);
        assert.equal(result.volatileClearTriggered, true);
        // after a clear only the freshly written graph remains
        assert.equal(result.volatileCacheSizeAfter, 1);
    });

    it('also clears when the table is already over capacity', () => {
        const { result } = shadowMiss(21);
        assert.equal(result.volatileClearTriggered, true);
        assert.equal(result.volatileCacheSizeAfter, 1);
    });

    it('never reports a clear on a cache hit, even with 20 volatile entries', () => {
        const snapshot = baseSnapshot({
            plugins,
            hasAnyShadow: true,
            volatileCacheKeys: Array.from({ length: 20 }, (_, i) => `old-key-${i}`),
        });
        snapshot.volatileCacheKeys[7] = expectedKey(snapshot);
        const result = diagnoseRenderGraphCache(snapshot);
        assert.equal(result.cacheHit, true);
        assert.equal(result.hitCacheKind, 'volatile');
        assert.equal(result.volatileClearTriggered, null);
        assert.equal(result.writeCacheKind, null);
    });

    it('never reports a volatile clear for a stable-cache write', () => {
        const snapshot = baseSnapshot({
            plugins,
            hasAnyShadow: false,
            volatileCacheKeys: Array.from({ length: 20 }, (_, i) => `old-key-${i}`),
        });
        const result = diagnoseRenderGraphCache(snapshot);
        assert.equal(result.writeCacheKind, 'stable');
        assert.equal(result.volatileClearTriggered, null);
        assert.equal(result.volatileCacheSizeAfter, 20);
    });
});

describe('diagnoseFrameSize', () => {
    function frameSnapshot(pixelRatio: number): FrameSizeSnapshot {
        return {
            width: 800,
            height: 600,
            pixelRatio,
            pluginNames: ['SceneClip', 'shadowMap', 'forward', 'composite'],
        };
    }

    it('keeps logical dimensions at pixelRatio 1 and notifies every plugin', () => {
        const result = diagnoseFrameSize(frameSnapshot(1));
        assert.equal(result.pixelWidth, 800);
        assert.equal(result.pixelHeight, 600);
        assert.deepEqual(result.notifiedPlugins, ['SceneClip', 'shadowMap', 'forward', 'composite']);
    });

    it('doubles pixel dimensions at pixelRatio 2', () => {
        const result = diagnoseFrameSize(frameSnapshot(2));
        assert.equal(result.pixelWidth, 1600);
        assert.equal(result.pixelHeight, 1200);
    });

    it('matches the setFrameSize multiplication for fractional ratios', () => {
        const result = diagnoseFrameSize(frameSnapshot(1.5));
        assert.equal(result.pixelWidth, 800 * 1.5);
        assert.equal(result.pixelHeight, 600 * 1.5);
    });

    it('returns a copy of the plugin list and JSON-serializable output', () => {
        const snapshot = frameSnapshot(2);
        const result = diagnoseFrameSize(snapshot);
        result.notifiedPlugins.push('mutated');
        assert.equal(snapshot.pluginNames.length, 4);
        assert.deepEqual(JSON.parse(JSON.stringify(result)), result);
    });
});
