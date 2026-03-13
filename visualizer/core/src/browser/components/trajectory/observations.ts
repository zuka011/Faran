import type { Visualizable } from "@/core/types";
import { write } from "@/utils/geometry";
import { noUpdater, type TraceUpdateCreator } from "./updater";

const ARROW_LENGTH = 0.3;

export const observationsUpdater: TraceUpdateCreator = (data, index) => {
    const observations = data.obstacles?.observations;

    if (!observations) {
        return noUpdater(data, index);
    }

    const maxObstacles = maxObstaclesIn(observations);
    const maxLineBufferLength =
        maxObstacles * write.headingLine.pointCount + Math.max(0, maxObstacles - 1);

    const pointBuffer = {
        x: new Array<number | null>(maxObstacles),
        y: new Array<number | null>(maxObstacles),
    };

    const arrowBuffer = {
        x: new Array<number | null>(maxLineBufferLength),
        y: new Array<number | null>(maxLineBufferLength),
    };

    const updateBuffers = (t: number) => {
        const obstacleCount = observations.x[t]?.length ?? 0;
        let pointCount = 0;
        let arrowOffset = 0;

        for (let i = 0; i < obstacleCount; i++) {
            const x = observations.x[t][i];
            const y = observations.y[t][i];
            const heading = observations.heading[t]?.[i];

            if (x == null || y == null || heading == null) {
                continue;
            }

            pointBuffer.x[pointCount] = x;
            pointBuffer.y[pointCount] = y;
            pointCount++;

            if (arrowOffset > 0) {
                arrowBuffer.x[arrowOffset] = null;
                arrowBuffer.y[arrowOffset] = null;
                arrowOffset++;
            }

            arrowOffset = write.headingLine(x, y, ARROW_LENGTH, heading, arrowBuffer, arrowOffset);
        }

        pointBuffer.x.length = pointCount;
        pointBuffer.y.length = pointCount;
        arrowBuffer.x.length = arrowOffset;
        arrowBuffer.y.length = arrowOffset;
    };

    return {
        createTemplates(theme) {
            void theme;
            updateBuffers(0);
            return [
                {
                    x: pointBuffer.x,
                    y: pointBuffer.y,
                    mode: "markers" as const,
                    marker: { color: "#3498db", size: 5, symbol: "circle" },
                    name: "Observation",
                    legendgroup: "observation",
                    showlegend: true,
                },
                {
                    x: arrowBuffer.x,
                    y: arrowBuffer.y,
                    mode: "lines" as const,
                    line: { color: "#2980b9", width: 2 },
                    name: "Observation",
                    legendgroup: "observation",
                    showlegend: false,
                },
            ];
        },

        updateTraces(timeStep) {
            updateBuffers(timeStep);
            return {
                data: [
                    pointBuffer as { x: number[]; y: number[] },
                    arrowBuffer as { x: number[]; y: number[] },
                ],
                updateIndices: [index, index + 1],
            };
        },
    };
};

function maxObstaclesIn(observations: Visualizable.ObstacleObservations): number {
    let max = 0;

    for (const timestep of observations.x) {
        max = Math.max(max, timestep.length);
    }

    return max;
}
