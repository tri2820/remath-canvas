import { useEffect, useMemo, useRef, useState } from "react";
import {
  type Body,
  type ForceBreakdown,
  type PhysicsAttraction,
  type PhysicsConstraint,
  type PhysicsRepulsor,
  calculateForces,
  connectedComponents,
  integrate,
  reconcileVelocities,
  solveConstraints,
} from "./physics";

type PointSpec = {
  id: string;
  label?: string;
  fixed?: boolean;
};

type CircleSpec = {
  id: string;
  type: "circle";
  name?: string;
  center: string;
  radius: number;
};

type LineSpec = {
  id: string;
  type: "line";
  through: [string, string];
};

type GraphEdgeSpec = {
  source: string;
  target: string;
  length?: number;
};

type GraphSpec = {
  id: string;
  type: "graph";
  name: string;
  nodes: string[];
  edges: GraphEdgeSpec[];
};

type ObjectSpec = CircleSpec | LineSpec | GraphSpec;

type OnConstraint = {
  type: "on";
  point: string;
  object: string;
};

type DistanceConstraint = {
  type: "distance";
  a: string;
  b: string;
  value: number;
};

type ConstraintSpec = OnConstraint | DistanceConstraint;

type GeometryDocument = {
  points: PointSpec[];
  objects: ObjectSpec[];
  constraints: ConstraintSpec[];
};

type World = {
  bodies: Map<string, Body>;
  initialized: boolean;
  width: number;
  height: number;
};

type DragState = {
  id: string;
  x: number;
  y: number;
  offsetX: number;
  offsetY: number;
};

type PhysicsSettings = {
  attraction: number;
  repulsion: number;
  damping: number;
};

const defaultPhysics: PhysicsSettings = {
  attraction: 30,
  repulsion: 220,
  damping: 6.3,
};

const circleDocument: GeometryDocument = {
  points: [
    { id: "O" },
    { id: "P" },
  ],
  objects: [
    { id: "c", type: "circle", name: "Circle", center: "O", radius: 100 },
  ],
  constraints: [
    { type: "on", point: "P", object: "c" },
  ],
};

const integratedCircleDocument: GeometryDocument = {
  points: [
    { id: "ic1:A", label: "A" },
    { id: "ic2:A", label: "A" },
    { id: "ic2:B", label: "B" },
    { id: "ic3:A", label: "A" },
    { id: "ic3:B", label: "B" },
    { id: "ic3:C", label: "C" },
    { id: "ic4:A", label: "A" },
    { id: "ic4:B", label: "B" },
    { id: "ic4:C", label: "C" },
    { id: "ic4:D", label: "D" },
    { id: "ic5:A", label: "A" },
    { id: "ic5:B", label: "B" },
    { id: "ic5:C", label: "C" },
    { id: "ic5:D", label: "D" },
    { id: "ic5:E", label: "E" },
    { id: "ic6:A", label: "A" },
    { id: "ic6:B", label: "B" },
    { id: "ic6:C", label: "C" },
    { id: "ic6:D", label: "D" },
    { id: "ic6:E", label: "E" },
    { id: "ic6:F", label: "F" },
  ],
  objects: [
    {
      id: "ic1",
      type: "graph",
      name: "Point",
      nodes: ["ic1:A"],
      edges: [],
    },
    {
      id: "ic2",
      type: "graph",
      name: "Digon",
      nodes: ["ic2:A", "ic2:B"],
      edges: [
        { source: "ic2:A", target: "ic2:B" },
        { source: "ic2:A", target: "ic2:B" },
      ],
    },
    {
      id: "ic3",
      type: "graph",
      name: "Doubled triangle",
      nodes: ["ic3:A", "ic3:B", "ic3:C"],
      edges: [
        { source: "ic3:A", target: "ic3:B" },
        { source: "ic3:A", target: "ic3:B" },
        { source: "ic3:B", target: "ic3:C" },
        { source: "ic3:B", target: "ic3:C" },
        { source: "ic3:C", target: "ic3:A" },
        { source: "ic3:C", target: "ic3:A" },
      ],
    },
    {
      id: "ic4",
      type: "graph",
      name: "Tetrahedron",
      nodes: ["ic4:A", "ic4:B", "ic4:C", "ic4:D"],
      edges: [
        { source: "ic4:A", target: "ic4:B" },
        { source: "ic4:A", target: "ic4:C" },
        { source: "ic4:A", target: "ic4:D" },
        { source: "ic4:B", target: "ic4:C" },
        { source: "ic4:B", target: "ic4:D" },
        { source: "ic4:C", target: "ic4:D" },
      ],
    },
    {
      id: "ic5",
      type: "graph",
      name: "4-simplex",
      nodes: ["ic5:A", "ic5:B", "ic5:C", "ic5:D", "ic5:E"],
      edges: [
        { source: "ic5:A", target: "ic5:B" },
        { source: "ic5:A", target: "ic5:C" },
        { source: "ic5:A", target: "ic5:D" },
        { source: "ic5:A", target: "ic5:E" },
        { source: "ic5:B", target: "ic5:C" },
        { source: "ic5:B", target: "ic5:D" },
        { source: "ic5:B", target: "ic5:E" },
        { source: "ic5:C", target: "ic5:D" },
        { source: "ic5:C", target: "ic5:E" },
        { source: "ic5:D", target: "ic5:E" },
      ],
    },
    {
      id: "ic6",
      type: "graph",
      name: "5-simplex",
      nodes: ["ic6:A", "ic6:B", "ic6:C", "ic6:D", "ic6:E", "ic6:F"],
      edges: [
        { source: "ic6:A", target: "ic6:B" },
        { source: "ic6:A", target: "ic6:C" },
        { source: "ic6:A", target: "ic6:D" },
        { source: "ic6:A", target: "ic6:E" },
        { source: "ic6:A", target: "ic6:F" },
        { source: "ic6:B", target: "ic6:C" },
        { source: "ic6:B", target: "ic6:D" },
        { source: "ic6:B", target: "ic6:E" },
        { source: "ic6:B", target: "ic6:F" },
        { source: "ic6:C", target: "ic6:D" },
        { source: "ic6:C", target: "ic6:E" },
        { source: "ic6:C", target: "ic6:F" },
        { source: "ic6:D", target: "ic6:E" },
        { source: "ic6:D", target: "ic6:F" },
        { source: "ic6:E", target: "ic6:F" },
      ],
    },
  ],
  constraints: [],
};

const allStructuresDocument: GeometryDocument = {
  points: [...circleDocument.points, ...integratedCircleDocument.points],
  objects: [...circleDocument.objects, ...integratedCircleDocument.objects],
  constraints: [...circleDocument.constraints, ...integratedCircleDocument.constraints],
};

function formatDocument(document: GeometryDocument) {
  return JSON.stringify(document, null, 2);
}

function parseDocument(source: string): GeometryDocument {
  const parsed: unknown = JSON.parse(source);
  if (!parsed || typeof parsed !== "object") {
    throw new Error("The root must be an object.");
  }

  const candidate = parsed as Partial<GeometryDocument>;
  if (
    !Array.isArray(candidate.points) ||
    !Array.isArray(candidate.objects) ||
    !Array.isArray(candidate.constraints)
  ) {
    throw new Error('Include "points", "objects", and "constraints" arrays.');
  }

  const points = candidate.points.map((point, index): PointSpec => {
    if (!point || typeof point !== "object" || typeof point.id !== "string") {
      throw new Error(`Point ${index + 1} needs a string id.`);
    }
    return {
      id: point.id,
      label: typeof point.label === "string" ? point.label : undefined,
      fixed: point.fixed === true,
    };
  });

  const pointIds = new Set(points.map((point) => point.id));
  if (pointIds.size !== points.length) throw new Error("Point ids must be unique.");

  const objects = candidate.objects.map((object, index): ObjectSpec => {
    if (!object || typeof object !== "object" || typeof object.id !== "string") {
      throw new Error(`Object ${index + 1} needs a string id.`);
    }

    if (object.type === "circle") {
      if (
        typeof object.center !== "string" ||
        !pointIds.has(object.center) ||
        typeof object.radius !== "number" ||
        object.radius <= 0
      ) {
        throw new Error(`Circle ${object.id} needs a valid center and positive radius.`);
      }
      return {
        id: object.id,
        type: "circle",
        name: typeof object.name === "string" ? object.name : undefined,
        center: object.center,
        radius: object.radius,
      };
    }

    if (object.type === "line") {
      if (
        !Array.isArray(object.through) ||
        object.through.length !== 2 ||
        object.through.some((id) => typeof id !== "string" || !pointIds.has(id))
      ) {
        throw new Error(`Line ${object.id} needs two valid points in "through".`);
      }
      return { id: object.id, type: "line", through: object.through as [string, string] };
    }

    if (object.type === "graph") {
      if (
        typeof object.name !== "string" ||
        !Array.isArray(object.nodes) ||
        object.nodes.some((id) => typeof id !== "string" || !pointIds.has(id)) ||
        !Array.isArray(object.edges)
      ) {
        throw new Error(`Graph ${object.id} needs a name, nodes, and edges.`);
      }
      const nodeIds = new Set(object.nodes);
      const edges = object.edges.map((edge, edgeIndex) => {
        if (
          !edge ||
          typeof edge.source !== "string" ||
          typeof edge.target !== "string" ||
          !nodeIds.has(edge.source) ||
          !nodeIds.has(edge.target) ||
          (edge.length !== undefined &&
            (typeof edge.length !== "number" || edge.length <= 0))
        ) {
          throw new Error(`Graph ${object.id} edge ${edgeIndex + 1} is invalid.`);
        }
        return edge.length === undefined
          ? { source: edge.source, target: edge.target }
          : { source: edge.source, target: edge.target, length: edge.length };
      });
      return {
        id: object.id,
        type: "graph",
        name: object.name,
        nodes: [...nodeIds],
        edges,
      };
    }

    throw new Error(`Object ${index + 1} has an unsupported type.`);
  });

  const objectIds = new Set(objects.map((object) => object.id));
  if (objectIds.size !== objects.length) throw new Error("Object ids must be unique.");
  const objectById = new Map(objects.map((object) => [object.id, object]));

  const constraints = candidate.constraints.map((constraint, index): ConstraintSpec => {
    if (!constraint || typeof constraint !== "object") {
      throw new Error(`Constraint ${index + 1} must be an object.`);
    }

    if (constraint.type === "on") {
      if (
        typeof constraint.point !== "string" ||
        !pointIds.has(constraint.point) ||
        typeof constraint.object !== "string" ||
        !objectIds.has(constraint.object) ||
        objectById.get(constraint.object)?.type === "graph"
      ) {
        throw new Error(`Constraint ${index + 1} references a missing point or object.`);
      }
      return { type: "on", point: constraint.point, object: constraint.object };
    }

    if (constraint.type === "distance") {
      if (
        typeof constraint.a !== "string" ||
        !pointIds.has(constraint.a) ||
        typeof constraint.b !== "string" ||
        !pointIds.has(constraint.b) ||
        typeof constraint.value !== "number" ||
        constraint.value <= 0
      ) {
        throw new Error(`Distance constraint ${index + 1} is invalid.`);
      }
      return {
        type: "distance",
        a: constraint.a,
        b: constraint.b,
        value: constraint.value,
      };
    }

    throw new Error(`Constraint ${index + 1} has unsupported type.`);
  });

  return { points, objects, constraints };
}

function resetWorld(world: World, document: GeometryDocument) {
  const bodies = new Map<string, Body>();
  const pointById = new Map(document.points.map((point) => [point.id, point]));
  const placed = new Set<string>();
  const place = (point: PointSpec, x: number, y: number) => {
    bodies.set(point.id, {
      id: point.id,
      x,
      y,
      vx: 0,
      vy: 0,
      fixed: point.fixed === true,
    });
    placed.add(point.id);
  };

  const circles = document.objects.filter((object): object is CircleSpec => object.type === "circle");
  const graphTop = circles.length
    ? Math.min(280, Math.max(230, world.height * 0.32))
    : 0;
  circles.forEach((circle, circleIndex) => {
    const centerPoint = pointById.get(circle.center);
    if (!centerPoint) return;
    const centerX = (circleIndex + 0.5) * world.width / circles.length;
    const centerY = graphTop / 2;
    place(centerPoint, centerX, centerY);

    const circumferencePoints = document.constraints
      .filter((constraint): constraint is OnConstraint =>
        constraint.type === "on" && constraint.object === circle.id)
      .map((constraint) => pointById.get(constraint.point))
      .filter((point): point is PointSpec => Boolean(point));
    circumferencePoints.forEach((point, pointIndex) => {
      const angle = Math.PI + pointIndex / Math.max(1, circumferencePoints.length) * Math.PI * 2;
      place(
        point,
        centerX + Math.cos(angle) * circle.radius,
        centerY + Math.sin(angle) * circle.radius,
      );
    });
  });

  const graphs = document.objects.filter((object): object is GraphSpec => object.type === "graph");
  if (graphs.length) {
    const graphHeight = Math.max(1, world.height - graphTop);
    const desiredColumns = Math.max(
      1,
      Math.min(graphs.length, Math.ceil(Math.sqrt(graphs.length * world.width / graphHeight))),
    );
    const columns = Math.min(
      desiredColumns,
      Math.max(1, Math.floor(world.width / 180)),
    );
    const rows = Math.ceil(graphs.length / columns);
    const cellWidth = world.width / columns;
    const cellHeight = graphHeight / rows;

    graphs.forEach((graph, graphIndex) => {
      const centerX = (graphIndex % columns + 0.5) * cellWidth;
      const centerY = graphTop + (Math.floor(graphIndex / columns) + 0.5) * cellHeight;
      const radius = Math.min(70, Math.min(cellWidth, cellHeight) * 0.24);
      graph.nodes.forEach((id, nodeIndex) => {
        const point = pointById.get(id);
        if (!point) return;
        const angle = -Math.PI / 2 + nodeIndex / Math.max(1, graph.nodes.length) * Math.PI * 2;
        const offset = graph.nodes.length === 1 ? 0 : radius;
        place(point, centerX + Math.cos(angle) * offset, centerY + Math.sin(angle) * offset);
      });
    });
  }

  const unplaced = document.points.filter((point) => !placed.has(point.id));
  const layoutRadius = Math.min(world.width, world.height - graphTop) * 0.18;
  unplaced.forEach((point, index) => {
    const angle = index / Math.max(1, unplaced.length) * Math.PI * 2;
    const offset = unplaced.length === 1 ? 0 : layoutRadius;
    place(
      point,
      world.width / 2 + Math.cos(angle) * offset,
      graphTop + (world.height - graphTop) / 2 + Math.sin(angle) * offset,
    );
  });
  world.bodies = bodies;
  world.initialized = true;
}

function objectDistance(
  object: ObjectSpec,
  x: number,
  y: number,
  bodies: Map<string, Body>,
) {
  if (object.type === "circle") {
    const center = bodies.get(object.center);
    if (!center) return 1;
    return Math.hypot(x - center.x, y - center.y) - object.radius;
  }
  if (object.type === "graph") return 1;

  const a = bodies.get(object.through[0]);
  const b = bodies.get(object.through[1]);
  if (!a || !b) return 1;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.max(0.0001, Math.hypot(dx, dy));
  return ((x - a.x) * dy - (y - a.y) * dx) / length;
}

function distanceEquation(
  aId: string,
  bId: string,
  target: number,
  stiffness = 1,
): PhysicsConstraint {
  return {
    particleIds: [aId, bId],
    stiffness,
    error: (bodies) => {
      const a = bodies.get(aId);
      const b = bodies.get(bId);
      if (!a || !b) return 0;
      return Math.hypot(b.x - a.x, b.y - a.y) - target;
    },
    gradient: (bodies, particleId) => {
      const a = bodies.get(aId);
      const b = bodies.get(bId);
      if (!a || !b) return { x: 0, y: 0 };
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = Math.max(0.0001, Math.hypot(dx, dy));
      const direction = particleId === aId ? -1 : 1;
      return { x: direction * dx / length, y: direction * dy / length };
    },
  };
}

function compilePhysicsConstraints(document: GeometryDocument): PhysicsConstraint[] {
  const objects = new Map(document.objects.map((object) => [object.id, object]));
  const equations = document.constraints.flatMap((constraint) => {
    if (constraint.type === "distance") {
      return [distanceEquation(constraint.a, constraint.b, constraint.value)];
    }

    const object = objects.get(constraint.object);
    if (!object) return [];
    if (object.type === "circle") {
      return [distanceEquation(object.center, constraint.point, object.radius)];
    }
    if (object.type === "graph") return [];

    const particleIds = [...new Set([constraint.point, ...object.through])];
    return [{
      particleIds,
      error: (bodies: Map<string, Body>) => {
        const point = bodies.get(constraint.point);
        if (!point) return 0;
        return objectDistance(object, point.x, point.y, bodies);
      },
    }];
  });

  const fixedEdges = document.objects.flatMap((object) => {
    if (object.type !== "graph") return [];
    return object.edges.flatMap((edge) =>
      edge.length === undefined
        ? []
        : [distanceEquation(edge.source, edge.target, edge.length)],
    );
  });

  return [...equations, ...fixedEdges];
}

function compilePhysicsAttractions(
  document: GeometryDocument,
  attractionStrength: number,
): PhysicsAttraction[] {
  return document.objects.flatMap((object) => {
    if (object.type !== "graph") return [];
    return object.edges.flatMap((edge) =>
      edge.length === undefined
        ? [{ a: edge.source, b: edge.target, strength: attractionStrength }]
        : [],
    );
  });
}

function compilePhysicsRepulsors(document: GeometryDocument): PhysicsRepulsor[] {
  const circles = document.objects.filter((object): object is CircleSpec => object.type === "circle");
  const circleCenters = new Set(circles.map((circle) => circle.center));
  const pointRepulsors: PhysicsRepulsor[] = document.points
    .filter((point) => !circleCenters.has(point.id))
    .map((point) => ({ anchorId: point.id, charge: 1 }));
  const circleRepulsors: PhysicsRepulsor[] = circles.map((circle) => ({
    anchorId: circle.center,
    charge: Math.max(1, circle.radius / 50),
    excludes: [
      circle.center,
      ...document.constraints
        .filter((constraint): constraint is OnConstraint =>
          constraint.type === "on" && constraint.object === circle.id)
        .map((constraint) => constraint.point),
    ],
  }));
  return [...pointRepulsors, ...circleRepulsors];
}

function componentBounds(
  component: Set<string>,
  document: GeometryDocument,
  bodies: Map<string, Body>,
) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  component.forEach((id) => {
    const body = bodies.get(id);
    if (!body) return;
    minX = Math.min(minX, body.x - 19);
    minY = Math.min(minY, body.y - 19);
    maxX = Math.max(maxX, body.x + 19);
    maxY = Math.max(maxY, body.y + 19);
  });

  document.objects.forEach((object) => {
    if (object.type !== "circle" || !component.has(object.center)) return;
    const center = bodies.get(object.center);
    if (!center) return;
    minX = Math.min(minX, center.x - object.radius);
    minY = Math.min(minY, center.y - object.radius - (object.name ? 24 : 0));
    maxX = Math.max(maxX, center.x + object.radius);
    maxY = Math.max(maxY, center.y + object.radius);
  });

  return { minX, minY, maxX, maxY };
}

function containStructures(
  components: Set<string>[],
  document: GeometryDocument,
  bodies: Map<string, Body>,
  width: number,
  height: number,
  draggedId: string | null,
) {
  const margin = 7;
  const corrections = new Map<string, { x: number; y: number }>();
  components.forEach((component) => {
    if (draggedId && component.has(draggedId)) return;
    const bounds = componentBounds(component, document, bodies);
    let correctionX = 0;
    let correctionY = 0;

    if (bounds.maxX - bounds.minX > width - margin * 2) {
      correctionX = width / 2 - (bounds.minX + bounds.maxX) / 2;
    } else if (bounds.minX < margin) {
      correctionX = margin - bounds.minX;
    } else if (bounds.maxX > width - margin) {
      correctionX = width - margin - bounds.maxX;
    }

    if (bounds.maxY - bounds.minY > height - margin * 2) {
      correctionY = height / 2 - (bounds.minY + bounds.maxY) / 2;
    } else if (bounds.minY < margin) {
      correctionY = margin - bounds.minY;
    } else if (bounds.maxY > height - margin) {
      correctionY = height - margin - bounds.maxY;
    }

    if (!correctionX && !correctionY) return;
    component.forEach((id) => {
      const body = bodies.get(id);
      if (!body || body.fixed) return;
      body.x += correctionX;
      body.y += correctionY;
      corrections.set(id, { x: correctionX, y: correctionY });
    });
  });
  return corrections;
}

function repelStructuresFromWalls(
  components: Set<string>[],
  document: GeometryDocument,
  bodies: Map<string, Body>,
  width: number,
  height: number,
  draggedId: string | null,
  repulsionStrength: number,
  deltaSeconds: number,
) {
  const softeningSquared = 20 * 20;
  const wallStrength = repulsionStrength * 100_000 * 0.008;
  components.forEach((component) => {
    if (draggedId && component.has(draggedId)) return;
    const bounds = componentBounds(component, document, bodies);
    const left = Math.max(0, bounds.minX);
    const right = Math.max(0, width - bounds.maxX);
    const top = Math.max(0, bounds.minY);
    const bottom = Math.max(0, height - bounds.maxY);
    const forceX = wallStrength / (left * left + softeningSquared)
      - wallStrength / (right * right + softeningSquared);
    const forceY = wallStrength / (top * top + softeningSquared)
      - wallStrength / (bottom * bottom + softeningSquared);
    component.forEach((id) => {
      const body = bodies.get(id);
      if (!body || body.fixed) return;
      body.vx += forceX * deltaSeconds;
      body.vy += forceY * deltaSeconds;
    });
  });
}

function crossing(
  ax: number,
  ay: number,
  av: number,
  bx: number,
  by: number,
  bv: number,
) {
  const denominator = av - bv;
  const t = Math.abs(denominator) < 0.00001 ? 0.5 : av / denominator;
  return { x: ax + (bx - ax) * t, y: ay + (by - ay) * t };
}

function drawImplicitContour(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  field: (x: number, y: number) => number,
) {
  const cell = 8;
  const trace = () => {
    context.beginPath();
    for (let y = 0; y < height; y += cell) {
      const bottom = Math.min(height, y + cell);
      for (let x = 0; x < width; x += cell) {
        const right = Math.min(width, x + cell);
        const values = [field(x, y), field(right, y), field(right, bottom), field(x, bottom)];
        const points: Array<{ x: number; y: number }> = [];

        if ((values[0] <= 0) !== (values[1] <= 0)) points.push(crossing(x, y, values[0], right, y, values[1]));
        if ((values[1] <= 0) !== (values[2] <= 0)) points.push(crossing(right, y, values[1], right, bottom, values[2]));
        if ((values[2] <= 0) !== (values[3] <= 0)) points.push(crossing(right, bottom, values[2], x, bottom, values[3]));
        if ((values[3] <= 0) !== (values[0] <= 0)) points.push(crossing(x, bottom, values[3], x, y, values[0]));

        for (let index = 0; index + 1 < points.length; index += 2) {
          context.moveTo(points[index].x, points[index].y);
          context.lineTo(points[index + 1].x, points[index + 1].y);
        }
      }
    }
    context.stroke();
  };

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#171814";
  context.lineWidth = 2;
  trace();
  context.restore();
}

function drawGrid(context: CanvasRenderingContext2D, width: number, height: number) {
  context.save();
  context.strokeStyle = "rgba(35, 36, 31, 0.07)";
  context.lineWidth = 1;
  context.beginPath();
  for (let x = 24; x < width; x += 24) {
    context.moveTo(x, 0);
    context.lineTo(x, height);
  }
  for (let y = 24; y < height; y += 24) {
    context.moveTo(0, y);
    context.lineTo(width, y);
  }
  context.stroke();
  context.restore();
}

function drawCircleLabels(
  context: CanvasRenderingContext2D,
  circles: CircleSpec[],
  bodies: Map<string, Body>,
) {
  context.save();
  context.fillStyle = "#171814";
  context.font = "600 10px 'DM Mono', monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  circles.forEach((circle) => {
    if (!circle.name) return;
    const center = bodies.get(circle.center);
    if (!center) return;
    context.fillText(circle.name, center.x, center.y - circle.radius - 16);
  });
  context.restore();
}

function drawGraphObjects(
  context: CanvasRenderingContext2D,
  graphs: GraphSpec[],
  bodies: Map<string, Body>,
) {
  graphs.forEach((graph) => {
    const totals = new Map<string, number>();
    graph.edges.forEach((edge) => {
      const key = [edge.source, edge.target].sort().join("::");
      totals.set(key, (totals.get(key) ?? 0) + 1);
    });
    const seen = new Map<string, number>();

    context.save();
    context.strokeStyle = "#171814";
    context.lineWidth = 1.8;
    context.lineCap = "round";
    graph.edges.forEach((edge) => {
      const source = bodies.get(edge.source);
      const target = bodies.get(edge.target);
      if (!source || !target) return;
      const key = [edge.source, edge.target].sort().join("::");
      const index = seen.get(key) ?? 0;
      seen.set(key, index + 1);
      const count = totals.get(key) ?? 1;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const length = Math.max(0.0001, Math.hypot(dx, dy));
      const offset = (index - (count - 1) / 2) * 8;
      const offsetX = -dy / length * offset;
      const offsetY = dx / length * offset;

      context.setLineDash(edge.length === undefined ? [] : [7, 6]);
      context.beginPath();
      context.moveTo(source.x + offsetX, source.y + offsetY);
      context.lineTo(target.x + offsetX, target.y + offsetY);
      context.stroke();
    });
    context.restore();

    const nodes = graph.nodes
      .map((id) => bodies.get(id))
      .filter((body): body is Body => Boolean(body));
    if (!nodes.length) return;
    const centerX = nodes.reduce((sum, body) => sum + body.x, 0) / nodes.length;
    const top = Math.min(...nodes.map((body) => body.y)) - 30;
    context.save();
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "#171814";
    context.font = "600 10px 'DM Mono', monospace";
    context.fillText(graph.name, centerX, top);
    context.restore();
  });
}

function drawArrow(
  context: CanvasRenderingContext2D,
  origin: Body,
  vector: { x: number; y: number },
  color: string,
  width: number,
) {
  const magnitude = Math.hypot(vector.x, vector.y);
  if (magnitude < 1) return;
  const displayLength = Math.min(36, magnitude * 0.0035);
  if (displayLength < 3) return;
  const unitX = vector.x / magnitude;
  const unitY = vector.y / magnitude;
  const endX = origin.x + unitX * displayLength;
  const endY = origin.y + unitY * displayLength;
  const head = 6;

  context.save();
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = width;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(origin.x, origin.y);
  context.lineTo(endX, endY);
  context.stroke();
  context.beginPath();
  context.moveTo(endX, endY);
  context.lineTo(
    endX - unitX * head - unitY * head * 0.55,
    endY - unitY * head + unitX * head * 0.55,
  );
  context.lineTo(
    endX - unitX * head + unitY * head * 0.55,
    endY - unitY * head - unitX * head * 0.55,
  );
  context.closePath();
  context.fill();
  context.restore();
}

function drawForceVectors(
  context: CanvasRenderingContext2D,
  bodies: Map<string, Body>,
  forces: Map<string, ForceBreakdown>,
  canvasWidth: number,
) {
  bodies.forEach((body, id) => {
    const force = forces.get(id);
    if (!force) return;
    drawArrow(context, body, force.attraction, "#d84a3a", 2);
    drawArrow(context, body, force.repulsion, "#2f6fce", 2);
    drawArrow(context, body, force.net, "#171814", 2.8);
  });

  const legendX = Math.max(12, canvasWidth - 212);
  const legendY = 18;
  const entries = [
    ["#d84a3a", "attraction"],
    ["#2f6fce", "repulsion"],
    ["#171814", "net"],
  ] as const;
  context.save();
  context.fillStyle = "rgba(247, 246, 239, 0.9)";
  context.fillRect(legendX - 8, legendY - 8, 208, 28);
  context.font = "600 9px 'DM Mono', monospace";
  context.textBaseline = "middle";
  let cursorX = legendX;
  entries.forEach(([color, label]) => {
    context.strokeStyle = color;
    context.lineWidth = color === "#171814" ? 2.8 : 2;
    context.beginPath();
    context.moveTo(cursorX, legendY + 5);
    context.lineTo(cursorX + 14, legendY + 5);
    context.stroke();
    context.fillStyle = "#171814";
    context.fillText(label, cursorX + 19, legendY + 5);
    cursorX += label === "attraction" ? 78 : label === "repulsion" ? 76 : 42;
  });
  context.restore();
}

function drawPoint(context: CanvasRenderingContext2D, body: Body, label: string) {
  context.save();
  context.shadowColor = "rgba(23, 24, 20, 0.16)";
  context.shadowBlur = 10;
  context.shadowOffsetY = 4;
  context.fillStyle = "#f7f6ef";
  context.strokeStyle = "#171814";
  context.lineWidth = 3;
  context.beginPath();
  context.arc(body.x, body.y, 19, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.restore();

  context.save();
  context.fillStyle = "#171814";
  context.font = "600 11px 'DM Mono', monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, body.x, body.y + 0.5);
  context.restore();
}

function drawConstraintLinks(
  context: CanvasRenderingContext2D,
  document: GeometryDocument,
  bodies: Map<string, Body>,
) {
  const objects = new Map(document.objects.map((object) => [object.id, object]));
  context.save();
  context.strokeStyle = "rgba(23, 24, 20, 0.5)";
  context.fillStyle = "#55574f";
  context.lineWidth = 1.4;
  context.setLineDash([7, 6]);
  context.font = "500 10px 'DM Mono', monospace";
  context.textAlign = "center";
  context.textBaseline = "bottom";

  document.constraints.forEach((constraint) => {
    let a: Body | undefined;
    let b: Body | undefined;
    let label = "";

    if (constraint.type === "distance") {
      a = bodies.get(constraint.a);
      b = bodies.get(constraint.b);
      label = String(constraint.value);
    } else {
      const object = objects.get(constraint.object);
      if (object?.type !== "circle") return;
      a = bodies.get(object.center);
      b = bodies.get(constraint.point);
    }

    if (!a || !b) return;
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
    if (label) context.fillText(label, (a.x + b.x) / 2, (a.y + b.y) / 2 - 5);
  });
  context.restore();
}

function PhysicsControl({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="physics-control">
      <span>
        <span>{label}</span>
        <output>{display}</output>
      </span>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onInput={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  );
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasPanelRef = useRef<HTMLDivElement>(null);
  const worldRef = useRef<World>({
    bodies: new Map(),
    initialized: false,
    width: 0,
    height: 0,
  });
  const dragRef = useRef<DragState | null>(null);
  const [document, setDocument] = useState(allStructuresDocument);
  const [draft, setDraft] = useState(() => formatDocument(allStructuresDocument));
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const [resetToken, setResetToken] = useState(0);
  const [physics, setPhysics] = useState(defaultPhysics);
  const [showVectors, setShowVectors] = useState(false);
  const [codeOpen, setCodeOpen] = useState(true);
  const [simulationOpen, setSimulationOpen] = useState(false);

  const physicsConstraints = useMemo(
    () => compilePhysicsConstraints(document),
    [document],
  );
  const physicsAttractions = useMemo(
    () => compilePhysicsAttractions(document, physics.attraction),
    [document, physics.attraction],
  );
  const physicsRepulsors = useMemo(
    () => compilePhysicsRepulsors(document),
    [document],
  );
  const graphObjects = useMemo(
    () => document.objects.filter((object): object is GraphSpec => object.type === "graph"),
    [document],
  );
  const pointLabels = useMemo(
    () => new Map(document.points.map((point) => [point.id, point.label ?? point.id])),
    [document],
  );

  const applyDraft = () => {
    try {
      const nextDocument = parseDocument(draft);
      setDocument(nextDocument);
      setDraft(formatDocument(nextDocument));
      setError("");
      worldRef.current.initialized = false;
      setResetToken((token) => token + 1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Invalid geometry code.");
    }
  };

  const formatDraft = () => {
    try {
      setDraft(formatDocument(parseDocument(draft)));
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Invalid geometry code.");
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const panel = canvasPanelRef.current;
    if (!canvas || !panel) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const world = worldRef.current;
    let animationFrame = 0;
    let previousTime = performance.now();

    const resize = () => {
      const bounds = panel.getBoundingClientRect();
      const width = Math.max(320, bounds.width);
      const height = Math.max(420, bounds.height);
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);

      if (!world.initialized) {
        world.width = width;
        world.height = height;
        resetWorld(world, document);
        solveConstraints(physicsConstraints, world.bodies, null);
      } else {
        const shiftX = (width - world.width) / 2;
        const shiftY = (height - world.height) / 2;
        world.bodies.forEach((body) => {
          body.x += shiftX;
          body.y += shiftY;
        });
        world.width = width;
        world.height = height;
      }
    };

    const observer = new ResizeObserver(resize);
    observer.observe(panel);
    resize();

    const draw = () => {
      context.clearRect(0, 0, world.width, world.height);
      context.fillStyle = "#f7f6ef";
      context.fillRect(0, 0, world.width, world.height);
      drawGrid(context, world.width, world.height);

      document.objects.forEach((object) => {
        if (object.type === "graph") return;
        drawImplicitContour(
          context,
          world.width,
          world.height,
          (x, y) => objectDistance(object, x, y, world.bodies),
        );
      });
      drawCircleLabels(
        context,
        document.objects.filter((object): object is CircleSpec => object.type === "circle"),
        world.bodies,
      );
      drawGraphObjects(context, graphObjects, world.bodies);
      if (showVectors) {
        const forceComponents = connectedComponents(
          physicsConstraints,
          physicsAttractions,
          world.bodies,
        );
        const forceBreakdowns = calculateForces(
          world.bodies,
          physicsAttractions,
          physicsRepulsors,
          forceComponents,
          physics.repulsion,
        );
        drawForceVectors(
          context,
          world.bodies,
          forceBreakdowns,
          world.width,
        );
      }
      drawConstraintLinks(context, document, world.bodies);
      world.bodies.forEach((body) => drawPoint(
        context,
        body,
        pointLabels.get(body.id) ?? body.id,
      ));
    };

    const tick = (time: number) => {
      const deltaSeconds = Math.min(1 / 30, Math.max(0, (time - previousTime) / 1000));
      previousTime = time;

      if (!paused) {
        const drag = dragRef.current;
        const components = connectedComponents(
          physicsConstraints,
          physicsAttractions,
          world.bodies,
        );
        const pointerTarget = drag
          ? {
              particleId: drag.id,
              x: drag.x + drag.offsetX,
              y: drag.y + drag.offsetY,
            }
          : null;
        const previousPositions = integrate(
          world.bodies,
          deltaSeconds,
          pointerTarget,
          physicsAttractions,
          physicsRepulsors,
          components,
          {
            repulsionStrength: physics.repulsion,
            damping: physics.damping,
          },
        );

        solveConstraints(physicsConstraints, world.bodies, drag?.id ?? null);

        const boundaryCorrections = containStructures(
          components,
          document,
          world.bodies,
          world.width,
          world.height,
          drag?.id ?? null,
        );

        reconcileVelocities(
          world.bodies,
          previousPositions,
          deltaSeconds,
          boundaryCorrections,
        );
        repelStructuresFromWalls(
          components,
          document,
          world.bodies,
          world.width,
          world.height,
          drag?.id ?? null,
          physics.repulsion,
          deltaSeconds,
        );
        if (drag) {
          const draggedBody = world.bodies.get(drag.id);
          if (draggedBody) {
            draggedBody.vx = 0;
            draggedBody.vy = 0;
          }
        }
      }
      draw();
      animationFrame = requestAnimationFrame(tick);
    };

    animationFrame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
    };
  }, [
    document,
    graphObjects,
    paused,
    physicsConstraints,
    physicsRepulsors,
    physicsAttractions,
    physics.repulsion,
    physics.damping,
    pointLabels,
    showVectors,
    resetToken,
  ]);

  const pointerPosition = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const point = pointerPosition(event);
    let nearest: Body | undefined;
    let nearestDistance = 30;
    worldRef.current.bodies.forEach((body) => {
      const distance = Math.hypot(point.x - body.x, point.y - body.y);
      if (distance <= nearestDistance) {
        nearest = body;
        nearestDistance = distance;
      }
    });
    if (!nearest || nearest.fixed) return;
    dragRef.current = {
      id: nearest.id,
      x: point.x,
      y: point.y,
      offsetX: nearest.x - point.x,
      offsetY: nearest.y - point.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const point = pointerPosition(event);
    drag.x = point.x;
    drag.y = point.y;
  };

  const handlePointerEnd = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (drag) {
      const point = pointerPosition(event);
      const body = worldRef.current.bodies.get(drag.id);
      if (body) {
        // Pointer-up can arrive before the next animation frame. Commit its
        // final position so a fast release never drops the last part of a drag.
        body.x = point.x + drag.offsetX;
        body.y = point.y + drag.offsetY;
        body.x = Math.max(26, Math.min(worldRef.current.width - 26, body.x));
        body.y = Math.max(26, Math.min(worldRef.current.height - 26, body.y));
        solveConstraints(physicsConstraints, worldRef.current.bodies, drag.id);
        body.vx = 0;
        body.vy = 0;
      }
    }
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const reset = () => {
    dragRef.current = null;
    worldRef.current.initialized = false;
    setResetToken((token) => token + 1);
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <header>
          <h1>ReMath</h1>
        </header>

        <section className={`code-section${codeOpen ? "" : " is-collapsed"}`}>
          <div className="section-heading">
            <button
              type="button"
              className="section-toggle"
              aria-expanded={codeOpen}
              onClick={() => setCodeOpen((open) => !open)}
            >
              <span>Code</span>
            </button>
            {codeOpen ? (
              <button type="button" className="text-button" onClick={formatDraft}>Format</button>
            ) : null}
          </div>
          {codeOpen ? (
            <>
              <div className="textarea-shell">
                <textarea
                  aria-label="Geometry code"
                  spellCheck={false}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                />
              </div>
              {error ? <p className="error-message">{error}</p> : null}
              <button type="button" className="apply-button" onClick={applyDraft}>Apply</button>
            </>
          ) : null}
        </section>

        <section className="physics-section">
          <div className="section-heading">
            <button
              type="button"
              className="section-toggle"
              aria-expanded={simulationOpen}
              onClick={() => setSimulationOpen((open) => !open)}
            >
              <span>Simulation</span>
            </button>
            {simulationOpen ? (
              <button
                type="button"
                className="text-button"
                onClick={() => setPhysics(defaultPhysics)}
              >
                Defaults
              </button>
            ) : null}
          </div>
          {simulationOpen ? (
            <>
              <div className="physics-controls">
                <PhysicsControl
                  label="Edge pull"
                  value={physics.attraction}
                  min={0}
                  max={80}
                  step={1}
                  display={String(physics.attraction)}
                  onChange={(attraction) => setPhysics((current) => ({ ...current, attraction }))}
                />
                <PhysicsControl
                  label="Node repulsion"
                  value={physics.repulsion}
                  min={0}
                  max={500}
                  step={5}
                  display={String(physics.repulsion)}
                  onChange={(repulsion) => setPhysics((current) => ({ ...current, repulsion }))}
                />
                <PhysicsControl
                  label="Damping"
                  value={physics.damping}
                  min={0.5}
                  max={14}
                  step={0.1}
                  display={physics.damping.toFixed(1)}
                  onChange={(damping) => setPhysics((current) => ({ ...current, damping }))}
                />
                <label className="vector-toggle">
                  <span>Force vectors</span>
                  <input
                    aria-label="Force vectors"
                    type="checkbox"
                    checked={showVectors}
                    onChange={(event) => setShowVectors(event.currentTarget.checked)}
                  />
                </label>
              </div>
              <div className="button-row">
                <button type="button" onClick={() => setPaused((current) => !current)}>
                  {paused ? "Resume" : "Pause"}
                </button>
                <button type="button" onClick={reset}>Reset</button>
              </div>
            </>
          ) : null}
        </section>
      </aside>

      <section className="canvas-panel" ref={canvasPanelRef}>
        <canvas
          ref={canvasRef}
          aria-label="ReMath geometry canvas"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerEnd}
          onPointerCancel={handlePointerEnd}
          onLostPointerCapture={handlePointerEnd}
        />
      </section>
    </main>
  );
}
