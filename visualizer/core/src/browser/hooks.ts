import { useEffect } from "preact/hooks";

/**
 * Dispatches a window resize event after the next animation frame,
 * allowing layout changes to settle before Plotly charts reflow.
 */
export function useResizeAfterLayoutChange(dependency: unknown): void {
    useEffect(() => {
        requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    }, [dependency]);
}
