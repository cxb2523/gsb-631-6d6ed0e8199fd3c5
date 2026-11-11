import type { RendererBackend } from '../renderer/IRenderer.js';

/**
 * Pure, side-effect-free diagnostics for {@link PostPipeline} render-graph
 * caching and frame-size scaling.
 *
 * This module never touches a WebGL context, never creates GPU resources and
 * never mutates global state. Every input is plain data (strings, numbers,
 * booleans, arrays) and every output is JSON serializable, so it can run in
 * tests without instantiating `WebGLRenderer`.
 *
 * The hash/cache branch logic mirrors `PostPipeline.createRenderGraph`
 * one-to-one:
 *
 *   key = backend + driven + occlusionCulling + effectiveMSAA
 *         + concat(PLUGIN_NAME + pluginHashFragment for every enabled plugin)
 *
 * where each boolean contributes `'1'` / `'0'`, exactly like
 * `HashKeyBuilder.bool` / `HashKeyBuilder.raw`.
 */

/**
 * Local copy of `RendererBackend.WEBGL_JS`. Kept as a string constant (instead
 * of a runtime import of the renderer module) so this file stays dependency
 * free and safe to evaluate outside a WebGL environment. The value must stay
 * in sync with the enum in `renderer/IRenderer.ts`.
 */
const WEBGL_JS_BACKEND = 'WEBGL_JS' as RendererBackend;

/** Number of volatile cache entries above which the whole table is cleared. */
export const VOLATILE_GRAPH_CACHE_LIMIT = 20;

/** Which graph cache table a graph is stored in. */
export type GraphCacheKind = 'stable' | 'volatile';

/** One enabled plugin as seen by the hash builder. */
export interface GraphCachePluginSnapshot {
    /** Value of `plugin.PLUGIN_NAME`, appended raw to the hash key first. */
    name: string;
    /**
     * Everything the plugin appends through its `updateGraphHash` callback
     * (e.g. a series of `'1'`/`'0'` from `hasher.bool(...)`, a raw number...).
     * Use an empty string for plugins whose `updateGraphHash` is a no-op.
     */
    hashFragment: string;
}

/** Immutable snapshot describing one `createRenderGraph` request. */
export interface RenderGraphSnapshot {
    /** `renderer.backend` for the current frame. */
    backend: RendererBackend;
    /** `renderingConfig.gpuDriven.enabled`. */
    gpuDrivenEnabled: boolean;
    /** `drivenCullingConfig.occlusionCullingEnabled`. */
    occlusionCullingEnabled: boolean;
    /**
     * Raw `renderingConfig.MSAA` value requested by the caller. On the
     * WEBGL_JS backend it is forced off and the effective value is reported.
     */
    msaa: boolean;
    /** Enabled plugins, in the same order used by `PostPipeline.plugins`. */
    plugins: GraphCachePluginSnapshot[];
    /**
     * Keys currently held by the stable cache (`graphCaches`).
     */
    stableCacheKeys: string[];
    /**
     * Keys currently held by the volatile cache (`volatileGraphCaches`,
     * shadow-only graphs).
     */
    volatileCacheKeys: string[];
    /**
     * Mirrors `adaptor.scene.scene.shaderComponentRegistry.light.hasAnyShadow`:
     * when true a newly built graph is committed to the volatile cache,
     * otherwise to the stable cache.
     */
    hasAnyShadow: boolean;
}

/** Structured, JSON-serializable result of a render-graph cache diagnosis. */
export interface RenderGraphDiagnostics {
    /** Final hash key, equal to the one `createRenderGraph` would build. */
    graphKey: string;
    /** Whether the key already exists in either cache table. */
    cacheHit: boolean;
    /** Table containing the key on a hit, `null` on a miss. */
    hitCacheKind: GraphCacheKind | null;
    /**
     * Raw requested MSAA value (the snapshot input).
     */
    requestedMSAA: boolean;
    /**
     * MSAA value actually hashed: forced to `false` on the WEBGL_JS backend.
     */
    effectiveMSAA: boolean;
    /** True when the requested MSAA was disabled because of the backend. */
    msaaInvalidated: boolean;
    /**
     * Table a newly built graph would be written to on a miss, `null` on a
     * hit (no write happens when the graph is reused).
     */
    writeCacheKind: GraphCacheKind | null;
    /**
     * On a volatile miss, whether committing the new entry clears the whole
     * volatile table first. The rule is "more than 20 entries after the
     * write", so 20 existing entries plus the new one (21) clears, while 19
     * plus one (exactly 20) does not. `null` when no volatile write happens.
     */
    volatileClearTriggered: boolean | null;
    /** Volatile table size before the hypothetical write. */
    volatileCacheSizeBefore: number;
    /** Volatile table size after the hypothetical write (post-clear size). */
    volatileCacheSizeAfter: number | null;
}

/** Immutable snapshot describing one `setFrameSize` call. */
export interface FrameSizeSnapshot {
    /** Logical width passed to `setFrameSize`. */
    width: number;
    /** Logical height passed to `setFrameSize`. */
    height: number;
    /** Value returned by `renderer.getPixelRatio()`. */
    pixelRatio: number;
    /**
     * Names (`PLUGIN_NAME`) of the plugins that will receive
     * `updateFrameSize`, in iteration order.
     */
    pluginNames: string[];
}

/** Structured, JSON-serializable result of a frame-size diagnosis. */
export interface FrameSizeDiagnostics {
    /** Logical width from the snapshot. */
    logicalWidth: number;
    /** Logical height from the snapshot. */
    logicalHeight: number;
    /** Pixel ratio from the snapshot. */
    pixelRatio: number;
    /** `width * pixelRatio`, equal to the width passed to `setSize`. */
    pixelWidth: number;
    /** `height * pixelRatio`, equal to the height passed to `setSize`. */
    pixelHeight: number;
    /** Plugins notified via `updateFrameSize(pixelWidth, pixelHeight)`. */
    notifiedPlugins: string[];
}

/**
 * Returned by `PostPipeline` audit methods after the pipeline has been
 * destroyed (`effectComposer` set to null). The pipeline cannot be diagnosed
 * in that state, so this object replaces the normal diagnostics result
 * instead of throwing.
 */
export interface PipelineDestroyedDiagnostics {
    diagnosticAvailable: false;
    reason: 'pipeline-destroyed';
}

/**
 * Build the exact same key string as `HashKeyBuilder` produces inside
 * `createRenderGraph`:
 *
 * `raw(backend).bool(driven).bool(occlusion).bool(msaa)` followed by, for
 * every enabled plugin, `raw(PLUGIN_NAME)` then the plugin's own fragment.
 */
export function buildRenderGraphKey(snapshot: RenderGraphSnapshot): string {
    const effectiveMSAA = snapshot.msaa && snapshot.backend !== WEBGL_JS_BACKEND;
    let key = `${snapshot.backend}${snapshot.gpuDrivenEnabled ? '1' : '0'}${
        snapshot.occlusionCullingEnabled ? '1' : '0'
    }${effectiveMSAA ? '1' : '0'}`;
    for (const plugin of snapshot.plugins) {
        key += plugin.name;
        key += plugin.hashFragment;
    }
    return key;
}

/**
 * Diagnose which cache branch `PostPipeline.createRenderGraph` would take for
 * the given request snapshot. Pure function: no WebGL, no GPU resources, no
 * global mutation, plain JSON-serializable output.
 */
export function diagnoseRenderGraphCache(snapshot: RenderGraphSnapshot): RenderGraphDiagnostics {
    const requestedMSAA = snapshot.msaa;
    const effectiveMSAA = requestedMSAA && snapshot.backend !== WEBGL_JS_BACKEND;
    const msaaInvalidated = requestedMSAA && !effectiveMSAA;

    const graphKey = buildRenderGraphKey(snapshot);

    const stableHit = snapshot.stableCacheKeys.includes(graphKey);
    const volatileHit = !stableHit && snapshot.volatileCacheKeys.includes(graphKey);
    const cacheHit = stableHit || volatileHit;
    const hitCacheKind: GraphCacheKind | null = stableHit ? 'stable' : volatileHit ? 'volatile' : null;

    const volatileCacheSizeBefore = snapshot.volatileCacheKeys.length;

    let writeCacheKind: GraphCacheKind | null = null;
    let volatileClearTriggered: boolean | null = null;
    let volatileCacheSizeAfter: number | null = null;

    if (!cacheHit) {
        // Mirrors the hasAnyShadow branch at the end of createRenderGraph.
        writeCacheKind = snapshot.hasAnyShadow ? 'volatile' : 'stable';

        if (writeCacheKind === 'volatile') {
            // Specification: clear when the table holds more than
            // VOLATILE_GRAPH_CACHE_LIMIT entries *after* the write, i.e. 21
            // entries trigger a clear, exactly 20 do not.
            const sizeAfterWrite = volatileCacheSizeBefore + 1;
            volatileClearTriggered = sizeAfterWrite > VOLATILE_GRAPH_CACHE_LIMIT;
            volatileCacheSizeAfter = volatileClearTriggered ? 1 : sizeAfterWrite;
        } else {
            volatileCacheSizeAfter = volatileCacheSizeBefore;
        }
    }

    return {
        graphKey,
        cacheHit,
        hitCacheKind,
        requestedMSAA,
        effectiveMSAA,
        msaaInvalidated,
        writeCacheKind,
        volatileClearTriggered,
        volatileCacheSizeBefore,
        volatileCacheSizeAfter,
    };
}

/**
 * Diagnose the pixel dimensions and plugin notifications produced by
 * `PostPipeline.setFrameSize`. The multiplication matches
 * `width * pixelRatio` / `height * pixelRatio` exactly; the returned plugin
 * list is a copy so callers cannot mutate the snapshot.
 */
export function diagnoseFrameSize(snapshot: FrameSizeSnapshot): FrameSizeDiagnostics {
    const pixelWidth = snapshot.width * snapshot.pixelRatio;
    const pixelHeight = snapshot.height * snapshot.pixelRatio;
    return {
        logicalWidth: snapshot.width,
        logicalHeight: snapshot.height,
        pixelRatio: snapshot.pixelRatio,
        pixelWidth,
        pixelHeight,
        notifiedPlugins: snapshot.pluginNames.slice(),
    };
}
