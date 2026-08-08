import { type ReactNode, useEffect, useId, useMemo, useRef, useState } from "react";
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
  name?: string;
  through: [string, string];
};

type SegmentSpec = {
  id: string;
  type: "segment";
  name?: string;
  between: [string, string];
};

type RaySpec = {
  id: string;
  type: "ray";
  name?: string;
  from: string;
  through: string;
};

type AngleSpec = {
  id: string;
  type: "angle";
  name?: string;
  arms: [string, string, string];
};

type PolygonSpec = {
  id: string;
  type: "polygon";
  name?: string;
  vertices: string[];
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

type ObjectSpec = CircleSpec | LineSpec | SegmentSpec | RaySpec | AngleSpec | PolygonSpec | GraphSpec;

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

type AngleConstraint = {
  type: "angle";
  object: string;
  value: number;
};

type ObjectRelationConstraint = {
  type: "parallel" | "perpendicular";
  objects: [string, string];
};

type ConstraintSpec = OnConstraint | DistanceConstraint | AngleConstraint | ObjectRelationConstraint;

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
  screenWidth: number;
  screenHeight: number;
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
  edgeRepulsion: number;
  extensionLength: number;
  damping: number;
};

const defaultPhysics: PhysicsSettings = {
  attraction: 30,
  repulsion: 220,
  edgeRepulsion: 32,
  extensionLength: 72,
  damping: 9.5,
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

const geometryPrimitivesDocument: GeometryDocument = {
  points: [
    { id: "line:A", label: "A" },
    { id: "line:B", label: "B" },
    { id: "line:P", label: "P" },
    { id: "parallel:A", label: "C" },
    { id: "parallel:B", label: "D" },
    { id: "perpendicular:A", label: "E" },
    { id: "perpendicular:B", label: "F" },
    { id: "segment:A", label: "A" },
    { id: "segment:B", label: "B" },
    { id: "ray:A", label: "A" },
    { id: "ray:B", label: "B" },
    { id: "angle:A", label: "A" },
    { id: "angle:V", label: "V" },
    { id: "angle:B", label: "B" },
    { id: "polygon:A", label: "A" },
    { id: "polygon:B", label: "B" },
    { id: "polygon:C", label: "C" },
    { id: "polygon:D", label: "D" },
    { id: "polygon:E", label: "E" },
  ],
  objects: [
    { id: "lineAB", type: "line", name: "Line AB", through: ["line:A", "line:B"] },
    {
      id: "segmentCD",
      type: "segment",
      name: "Segment CD",
      between: ["parallel:A", "parallel:B"],
    },
    {
      id: "segmentEF",
      type: "segment",
      name: "Segment EF",
      between: ["perpendicular:A", "perpendicular:B"],
    },
    { id: "segment", type: "segment", name: "Segment", between: ["segment:A", "segment:B"] },
    { id: "ray", type: "ray", name: "Ray", from: "ray:A", through: "ray:B" },
    { id: "angle", type: "angle", name: "Angle", arms: ["angle:A", "angle:V", "angle:B"] },
    {
      id: "polygon",
      type: "polygon",
      name: "Pentagon",
      vertices: ["polygon:A", "polygon:B", "polygon:C", "polygon:D", "polygon:E"],
    },
  ],
  constraints: [
    { type: "on", point: "line:P", object: "lineAB" },
    { type: "parallel", objects: ["lineAB", "segmentCD"] },
    { type: "perpendicular", objects: ["lineAB", "segmentEF"] },
    { type: "angle", object: "angle", value: 60 },
  ],
};

const allStructuresDocument: GeometryDocument = {
  points: [
    ...circleDocument.points,
    ...integratedCircleDocument.points,
    ...geometryPrimitivesDocument.points,
  ],
  objects: [
    ...circleDocument.objects,
    ...integratedCircleDocument.objects,
    ...geometryPrimitivesDocument.objects,
  ],
  constraints: [
    ...circleDocument.constraints,
    ...integratedCircleDocument.constraints,
    ...geometryPrimitivesDocument.constraints,
  ],
};

function formatDocument(document: GeometryDocument) {
  return JSON.stringify(document, null, 2);
}

function objectPointIds(object: ObjectSpec): string[] {
  if (object.type === "circle") return [object.center];
  if (object.type === "line") return object.through;
  if (object.type === "segment") return object.between;
  if (object.type === "ray") return [object.from, object.through];
  if (object.type === "angle") return object.arms;
  if (object.type === "polygon") return object.vertices;
  return object.nodes;
}

function lineEndpoints(object: ObjectSpec): [string, string] | undefined {
  if (object.type === "line") return object.through;
  if (object.type === "segment") return object.between;
  if (object.type === "ray") return [object.from, object.through];
  return undefined;
}

function objectName(object: ObjectSpec) {
  return object.type === "graph" ? object.name : object.name;
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
      ...(typeof point.label === "string" ? { label: point.label } : {}),
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
      return {
        id: object.id,
        type: "line",
        name: typeof object.name === "string" ? object.name : undefined,
        through: object.through as [string, string],
      };
    }

    if (object.type === "segment") {
      if (
        !Array.isArray(object.between) ||
        object.between.length !== 2 ||
        object.between.some((id) => typeof id !== "string" || !pointIds.has(id))
      ) {
        throw new Error(`Segment ${object.id} needs two valid points in "between".`);
      }
      return {
        id: object.id,
        type: "segment",
        name: typeof object.name === "string" ? object.name : undefined,
        between: object.between as [string, string],
      };
    }

    if (object.type === "ray") {
      if (
        typeof object.from !== "string" ||
        !pointIds.has(object.from) ||
        typeof object.through !== "string" ||
        !pointIds.has(object.through) ||
        object.from === object.through
      ) {
        throw new Error(`Ray ${object.id} needs valid distinct "from" and "through" points.`);
      }
      return {
        id: object.id,
        type: "ray",
        name: typeof object.name === "string" ? object.name : undefined,
        from: object.from,
        through: object.through,
      };
    }

    if (object.type === "angle") {
      if (
        !Array.isArray(object.arms) ||
        object.arms.length !== 3 ||
        object.arms.some((id) => typeof id !== "string" || !pointIds.has(id)) ||
        new Set(object.arms).size !== 3
      ) {
        throw new Error(`Angle ${object.id} needs three distinct valid points in "arms".`);
      }
      return {
        id: object.id,
        type: "angle",
        name: typeof object.name === "string" ? object.name : undefined,
        arms: object.arms as [string, string, string],
      };
    }

    if (object.type === "polygon") {
      if (
        !Array.isArray(object.vertices) ||
        object.vertices.length < 3 ||
        object.vertices.some((id) => typeof id !== "string" || !pointIds.has(id)) ||
        new Set(object.vertices).size !== object.vertices.length
      ) {
        throw new Error(`Polygon ${object.id} needs at least three distinct valid vertices.`);
      }
      return {
        id: object.id,
        type: "polygon",
        name: typeof object.name === "string" ? object.name : undefined,
        vertices: object.vertices,
      };
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
      const target = typeof constraint.object === "string"
        ? objectById.get(constraint.object)
        : undefined;
      if (
        typeof constraint.point !== "string" ||
        !pointIds.has(constraint.point) ||
        typeof constraint.object !== "string" ||
        !target ||
        (target.type !== "circle" && target.type !== "line")
      ) {
        throw new Error(`On constraint ${index + 1} needs a point and a circle or line.`);
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

    if (constraint.type === "angle") {
      const object = typeof constraint.object === "string"
        ? objectById.get(constraint.object)
        : undefined;
      if (
        object?.type !== "angle" ||
        typeof constraint.value !== "number" ||
        constraint.value <= 0 ||
        constraint.value >= 180
      ) {
        throw new Error(`Angle constraint ${index + 1} needs an angle and value between 0 and 180.`);
      }
      return { type: "angle", object: constraint.object, value: constraint.value };
    }

    if (constraint.type === "parallel" || constraint.type === "perpendicular") {
      if (
        !Array.isArray(constraint.objects) ||
        constraint.objects.length !== 2 ||
        constraint.objects.some((id) => typeof id !== "string")
      ) {
        throw new Error(`${constraint.type} constraint ${index + 1} needs two object ids.`);
      }
      const first = objectById.get(constraint.objects[0]);
      const second = objectById.get(constraint.objects[1]);
      if (
        !first ||
        !second ||
        first.id === second.id ||
        !lineEndpoints(first) ||
        !lineEndpoints(second)
      ) {
        throw new Error(`${constraint.type} constraint ${index + 1} needs two line-like objects.`);
      }
      return {
        type: constraint.type,
        objects: constraint.objects as [string, string],
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

  const availablePoints = document.points.filter((point) => !placed.has(point.id));
  const availableIds = new Set(availablePoints.map((point) => point.id));
  const parents = new Map(availablePoints.map((point) => [point.id, point.id]));
  const find = (id: string): string => {
    const parent = parents.get(id);
    if (!parent || parent === id) return id;
    const root = find(parent);
    parents.set(id, root);
    return root;
  };
  const union = (ids: string[]) => {
    const members = ids.filter((id) => availableIds.has(id));
    if (members.length < 2) return;
    const root = find(members[0]);
    members.slice(1).forEach((id) => parents.set(find(id), root));
  };

  document.objects.forEach((object) => {
    if (object.type !== "circle") union(objectPointIds(object));
  });
  const objectById = new Map(document.objects.map((object) => [object.id, object]));
  document.constraints.forEach((constraint) => {
    if (constraint.type === "distance") union([constraint.a, constraint.b]);
    if (constraint.type === "on") {
      const object = objectById.get(constraint.object);
      if (object?.type === "line") union([constraint.point, ...object.through]);
    }
    if (constraint.type === "parallel" || constraint.type === "perpendicular") {
      const first = objectById.get(constraint.objects[0]);
      const second = objectById.get(constraint.objects[1]);
      if (first && second) union([...objectPointIds(first), ...objectPointIds(second)]);
    }
  });

  const groups = new Map<string, PointSpec[]>();
  availablePoints.forEach((point) => {
    const root = find(point.id);
    const group = groups.get(root) ?? [];
    group.push(point);
    groups.set(root, group);
  });
  const layoutUnits = [...groups.values()];
  if (layoutUnits.length) {
    const graphHeight = Math.max(1, world.height - graphTop);
    const desiredColumns = Math.max(
      1,
      Math.min(
        layoutUnits.length,
        Math.ceil(Math.sqrt(layoutUnits.length * world.width / graphHeight)),
      ),
    );
    const columns = Math.min(
      desiredColumns,
      Math.max(1, Math.floor(world.width / 180)),
    );
    const rows = Math.ceil(layoutUnits.length / columns);
    const cellWidth = world.width / columns;
    const cellHeight = graphHeight / rows;

    layoutUnits.forEach((points, unitIndex) => {
      const centerX = (unitIndex % columns + 0.5) * cellWidth;
      const centerY = graphTop + (Math.floor(unitIndex / columns) + 0.5) * cellHeight;
      const radius = Math.min(70, Math.min(cellWidth, cellHeight) * 0.24);
      points.forEach((point, pointIndex) => {
        const angle = -Math.PI / 2 + pointIndex / Math.max(1, points.length) * Math.PI * 2;
        const offset = points.length === 1 ? 0 : radius;
        place(point, centerX + Math.cos(angle) * offset, centerY + Math.sin(angle) * offset);
      });
    });
  }
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
  if (object.type !== "line") return 1;

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

function segmentCrossesLineEquation(
  line: [string, string],
  segment: [string, string],
): PhysicsConstraint {
  return {
    particleIds: [...line, ...segment],
    connects: false,
    error: (bodies) => {
      const lineStart = bodies.get(line[0]);
      const lineEnd = bodies.get(line[1]);
      const segmentStart = bodies.get(segment[0]);
      const segmentEnd = bodies.get(segment[1]);
      if (!lineStart || !lineEnd || !segmentStart || !segmentEnd) return 0;
      const dx = lineEnd.x - lineStart.x;
      const dy = lineEnd.y - lineStart.y;
      const length = Math.max(0.0001, Math.hypot(dx, dy));
      const signedDistance = (point: Body) =>
        ((point.x - lineStart.x) * dy - (point.y - lineStart.y) * dx) / length;
      const product = signedDistance(segmentStart) * signedDistance(segmentEnd);
      return product > 0 ? Math.sqrt(product) : 0;
    },
  };
}

function compilePhysicsConstraints(document: GeometryDocument): PhysicsConstraint[] {
  const objects = new Map(document.objects.map((object) => [object.id, object]));
  const equations = document.constraints.flatMap((constraint) => {
    if (constraint.type === "distance") {
      return [distanceEquation(constraint.a, constraint.b, constraint.value)];
    }

    if (constraint.type === "angle") {
      const object = objects.get(constraint.object);
      if (object?.type !== "angle") return [];
      const [firstId, vertexId, secondId] = object.arms;
      const target = constraint.value * Math.PI / 180;
      return [{
        particleIds: object.arms,
        error: (bodies: Map<string, Body>) => {
          const first = bodies.get(firstId);
          const vertex = bodies.get(vertexId);
          const second = bodies.get(secondId);
          if (!first || !vertex || !second) return 0;
          const firstX = first.x - vertex.x;
          const firstY = first.y - vertex.y;
          const secondX = second.x - vertex.x;
          const secondY = second.y - vertex.y;
          const denominator = Math.max(
            0.0001,
            Math.hypot(firstX, firstY) * Math.hypot(secondX, secondY),
          );
          const cosine = Math.max(
            -1,
            Math.min(1, (firstX * secondX + firstY * secondY) / denominator),
          );
          return Math.acos(cosine) - target;
        },
      }];
    }

    if (constraint.type === "parallel" || constraint.type === "perpendicular") {
      const first = objects.get(constraint.objects[0]);
      const second = objects.get(constraint.objects[1]);
      if (!first || !second) return [];
      const firstEnds = lineEndpoints(first);
      const secondEnds = lineEndpoints(second);
      if (!firstEnds || !secondEnds) return [];
      const orientation: PhysicsConstraint = {
        particleIds: [...firstEnds, ...secondEnds],
        connects: false,
        error: (bodies: Map<string, Body>) => {
          const firstA = bodies.get(firstEnds[0]);
          const firstB = bodies.get(firstEnds[1]);
          const secondA = bodies.get(secondEnds[0]);
          const secondB = bodies.get(secondEnds[1]);
          if (!firstA || !firstB || !secondA || !secondB) return 0;
          const firstX = firstB.x - firstA.x;
          const firstY = firstB.y - firstA.y;
          const secondX = secondB.x - secondA.x;
          const secondY = secondB.y - secondA.y;
          const denominator = Math.max(
            0.0001,
            Math.hypot(firstX, firstY) * Math.hypot(secondX, secondY),
          );
          return constraint.type === "parallel"
            ? (firstX * secondY - firstY * secondX) / denominator
            : (firstX * secondX + firstY * secondY) / denominator;
        },
      };
      const boundedIntersection = constraint.type !== "perpendicular"
        ? []
        : first.type === "line" && second.type === "segment"
          ? [segmentCrossesLineEquation(firstEnds, secondEnds)]
          : first.type === "segment" && second.type === "line"
            ? [segmentCrossesLineEquation(secondEnds, firstEnds)]
            : [];
      return [orientation, ...boundedIntersection];
    }

    if (constraint.type !== "on") return [];
    const object = objects.get(constraint.object);
    if (!object) return [];
    if (object.type === "circle") {
      return [distanceEquation(object.center, constraint.point, object.radius)];
    }
    if (object.type !== "line") return [];

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
    if (object.type === "graph") {
      return object.edges.flatMap((edge) =>
        edge.length === undefined
          ? [{ a: edge.source, b: edge.target, strength: attractionStrength }]
          : [],
      );
    }
    const ends = lineEndpoints(object);
    if (ends) return [{ a: ends[0], b: ends[1], strength: attractionStrength }];
    if (object.type === "angle") {
      return [
        { a: object.arms[1], b: object.arms[0], strength: attractionStrength },
        { a: object.arms[1], b: object.arms[2], strength: attractionStrength },
      ];
    }
    if (object.type === "polygon") {
      return object.vertices.map((vertex, index) => ({
        a: vertex,
        b: object.vertices[(index + 1) % object.vertices.length],
        strength: attractionStrength,
      }));
    }
    return [];
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
      if (!body) return;
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
      if (!body) return;
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
  strokeStyle = "#171814",
  lineWidth = 2,
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
  context.strokeStyle = strokeStyle;
  context.lineWidth = lineWidth;
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

const OPEN_END_MARGIN = 14;
type CanvasPoint = { x: number; y: number };

function visibleExtensionLength(
  point: CanvasPoint,
  directionX: number,
  directionY: number,
  width: number,
  height: number,
  maximumLength: number,
) {
  let available = maximumLength;
  if (directionX > 0.0001) {
    available = Math.min(available, (width - OPEN_END_MARGIN - point.x) / directionX);
  } else if (directionX < -0.0001) {
    available = Math.min(available, (OPEN_END_MARGIN - point.x) / directionX);
  }
  if (directionY > 0.0001) {
    available = Math.min(available, (height - OPEN_END_MARGIN - point.y) / directionY);
  } else if (directionY < -0.0001) {
    available = Math.min(available, (OPEN_END_MARGIN - point.y) / directionY);
  }
  return Math.max(0, available);
}

function openLineGeometry(
  start: CanvasPoint,
  end: CanvasPoint,
  openAtStart: boolean,
  width: number,
  height: number,
  extensionLength: number,
) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.max(0.0001, Math.hypot(dx, dy));
  const unitX = dx / length;
  const unitY = dy / length;
  const startExtension = openAtStart
    ? visibleExtensionLength(start, -unitX, -unitY, width, height, extensionLength)
    : 0;
  const endExtension = visibleExtensionLength(
    end,
    unitX,
    unitY,
    width,
    height,
    extensionLength,
  );
  const renderedStart = {
    x: start.x - unitX * startExtension,
    y: start.y - unitY * startExtension,
  };
  const renderedEnd = {
    x: end.x + unitX * endExtension,
    y: end.y + unitY * endExtension,
  };
  const renderedLength = length + startExtension + endExtension;
  return { renderedStart, renderedEnd, renderedLength, startExtension, length };
}

function renderOpenLine(
  context: CanvasRenderingContext2D,
  start: CanvasPoint,
  end: CanvasPoint,
  openAtStart: boolean,
  width: number,
  height: number,
  extensionLength: number,
) {
  const {
    renderedStart,
    renderedEnd,
    renderedLength,
    startExtension,
    length,
  } = openLineGeometry(start, end, openAtStart, width, height, extensionLength);
  const gradient = context.createLinearGradient(
    renderedStart.x,
    renderedStart.y,
    renderedEnd.x,
    renderedEnd.y,
  );

  if (openAtStart) {
    gradient.addColorStop(0, "rgba(23, 24, 20, 0)");
    gradient.addColorStop(startExtension / renderedLength, "rgba(23, 24, 20, 0.55)");
  } else {
    gradient.addColorStop(0, "rgba(23, 24, 20, 0.55)");
  }
  gradient.addColorStop(
    (startExtension + length) / renderedLength,
    "rgba(23, 24, 20, 0.55)",
  );
  gradient.addColorStop(1, "rgba(23, 24, 20, 0)");

  context.save();
  context.strokeStyle = gradient;
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(renderedStart.x, renderedStart.y);
  context.lineTo(renderedEnd.x, renderedEnd.y);
  context.stroke();
  context.restore();
}

function supportingLineIntersection(
  first: { start: CanvasPoint; end: CanvasPoint },
  second: { start: CanvasPoint; end: CanvasPoint },
) {
  const firstDirection = {
    x: first.end.x - first.start.x,
    y: first.end.y - first.start.y,
  };
  const secondDirection = {
    x: second.end.x - second.start.x,
    y: second.end.y - second.start.y,
  };
  const denominator = firstDirection.x * secondDirection.y
    - firstDirection.y * secondDirection.x;
  if (Math.abs(denominator) < 0.0001) return undefined;
  const offsetX = second.start.x - first.start.x;
  const offsetY = second.start.y - first.start.y;
  const scale = (offsetX * secondDirection.y - offsetY * secondDirection.x) / denominator;
  return {
    x: first.start.x + firstDirection.x * scale,
    y: first.start.y + firstDirection.y * scale,
  };
}

function lineVisualEndpoints(
  line: LineSpec,
  document: GeometryDocument,
  bodies: Map<string, Body>,
) {
  const start = bodies.get(line.through[0]);
  const end = bodies.get(line.through[1]);
  if (!start || !end) return undefined;
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.max(0.0001, Math.hypot(dx, dy));
  const unitX = dx / length;
  const unitY = dy / length;
  const anchors: CanvasPoint[] = [start, end];
  const objects = new Map(document.objects.map((object) => [object.id, object]));

  document.constraints.forEach((constraint) => {
    if (constraint.type === "on" && constraint.object === line.id) {
      const point = bodies.get(constraint.point);
      if (point) anchors.push(point);
      return;
    }
    if (constraint.type !== "perpendicular" || !constraint.objects.includes(line.id)) return;
    const otherId = constraint.objects[0] === line.id
      ? constraint.objects[1]
      : constraint.objects[0];
    const other = objects.get(otherId);
    const ends = other ? lineEndpoints(other) : undefined;
    if (!ends) return;
    const otherStart = bodies.get(ends[0]);
    const otherEnd = bodies.get(ends[1]);
    if (!otherStart || !otherEnd) return;
    const intersection = supportingLineIntersection(
      { start, end },
      { start: otherStart, end: otherEnd },
    );
    if (intersection) anchors.push(intersection);
  });

  const projections = anchors.map(
    (point) => (point.x - start.x) * unitX + (point.y - start.y) * unitY,
  );
  const minimum = Math.min(...projections);
  const maximum = Math.max(...projections);
  return {
    start: { x: start.x + unitX * minimum, y: start.y + unitY * minimum },
    end: { x: start.x + unitX * maximum, y: start.y + unitY * maximum },
  };
}

function primitiveEdgePaths(
  document: GeometryDocument,
  bodies: Map<string, Body>,
  width: number,
  height: number,
  extensionLength: number,
) {
  const paths: EdgePath[] = [];
  const add = (sourceId: string, targetId: string) => {
    const source = bodies.get(sourceId);
    const target = bodies.get(targetId);
    if (!source || !target) return;
    paths.push({ edge: { source: sourceId, target: targetId }, source, target });
  };
  document.objects.forEach((object) => {
    if (object.type === "line") {
      const visual = lineVisualEndpoints(object, document, bodies);
      if (!visual) return;
      const { renderedStart, renderedEnd } = openLineGeometry(
        visual.start,
        visual.end,
        true,
        width,
        height,
        extensionLength,
      );
      paths.push({
        edge: { source: object.through[0], target: object.through[1] },
        source: renderedStart,
        target: renderedEnd,
      });
    }
    if (object.type === "segment") add(...object.between);
    if (object.type === "ray") {
      const start = bodies.get(object.from);
      const through = bodies.get(object.through);
      if (!start || !through) return;
      const { renderedStart, renderedEnd } = openLineGeometry(
        start,
        through,
        false,
        width,
        height,
        extensionLength,
      );
      paths.push({
        edge: { source: object.from, target: object.through },
        source: renderedStart,
        target: renderedEnd,
      });
    }
    if (object.type === "angle") {
      add(object.arms[1], object.arms[0]);
      add(object.arms[1], object.arms[2]);
    }
    if (object.type === "polygon") {
      object.vertices.forEach((vertex, index) => {
        add(vertex, object.vertices[(index + 1) % object.vertices.length]);
      });
    }
  });
  return paths;
}

function drawPrimitiveObjects(
  context: CanvasRenderingContext2D,
  document: GeometryDocument,
  bodies: Map<string, Body>,
  width: number,
  height: number,
  extensionLength: number,
) {
  const angleValues = new Map(
    document.constraints.flatMap((constraint) =>
      constraint.type === "angle" ? [[constraint.object, constraint.value] as const] : []),
  );
  context.save();
  context.strokeStyle = "#171814";
  context.fillStyle = "#171814";
  context.lineWidth = 1.8;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.font = "500 9px 'DM Mono', monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";

  document.objects.forEach((object) => {
    if (object.type === "line") {
      const visual = lineVisualEndpoints(object, document, bodies);
      if (!visual) return;
      renderOpenLine(
        context,
        visual.start,
        visual.end,
        true,
        width,
        height,
        extensionLength,
      );
    }

    if (object.type === "segment") {
      const start = bodies.get(object.between[0]);
      const end = bodies.get(object.between[1]);
      if (!start || !end) return;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }

    if (object.type === "ray") {
      const start = bodies.get(object.from);
      const through = bodies.get(object.through);
      if (!start || !through) return;
      renderOpenLine(context, start, through, false, width, height, extensionLength);
    }

    if (object.type === "polygon") {
      const vertices = object.vertices
        .map((id) => bodies.get(id))
        .filter((body): body is Body => Boolean(body));
      if (vertices.length !== object.vertices.length) return;
      context.beginPath();
      context.moveTo(vertices[0].x, vertices[0].y);
      vertices.slice(1).forEach((vertex) => context.lineTo(vertex.x, vertex.y));
      context.closePath();
      context.stroke();
    }

    if (object.type === "angle") {
      const first = bodies.get(object.arms[0]);
      const vertex = bodies.get(object.arms[1]);
      const second = bodies.get(object.arms[2]);
      if (!first || !vertex || !second) return;
      context.beginPath();
      context.moveTo(first.x, first.y);
      context.lineTo(vertex.x, vertex.y);
      context.lineTo(second.x, second.y);
      context.stroke();

      const firstAngle = Math.atan2(first.y - vertex.y, first.x - vertex.x);
      const secondAngle = Math.atan2(second.y - vertex.y, second.x - vertex.x);
      let sweep = secondAngle - firstAngle;
      while (sweep > Math.PI) sweep -= Math.PI * 2;
      while (sweep < -Math.PI) sweep += Math.PI * 2;
      const radius = Math.min(
        24,
        Math.hypot(first.x - vertex.x, first.y - vertex.y) * 0.35,
        Math.hypot(second.x - vertex.x, second.y - vertex.y) * 0.35,
      );
      context.beginPath();
      context.arc(vertex.x, vertex.y, radius, firstAngle, firstAngle + sweep, sweep < 0);
      context.stroke();

      const value = angleValues.get(object.id);
      if (value !== undefined) {
        const middle = firstAngle + sweep / 2;
        context.fillText(
          `${value}°`,
          vertex.x + Math.cos(middle) * (radius + 12),
          vertex.y + Math.sin(middle) * (radius + 12),
        );
      }
    }
  });
  context.restore();
}

function drawRelationMarkers(
  context: CanvasRenderingContext2D,
  document: GeometryDocument,
  bodies: Map<string, Body>,
) {
  const objects = new Map(document.objects.map((object) => [object.id, object]));
  const bodyEnds = (objectId: string) => {
    const object = objects.get(objectId);
    const ends = object ? lineEndpoints(object) : undefined;
    if (!ends) return undefined;
    const start = bodies.get(ends[0]);
    const end = bodies.get(ends[1]);
    return start && end ? { start, end } : undefined;
  };
  const direction = ({ start, end }: { start: Body; end: Body }) => {
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.max(0.0001, Math.hypot(dx, dy));
    return { x: dx / length, y: dy / length };
  };

  context.save();
  context.strokeStyle = "#55574f";
  context.lineWidth = 1.3;
  context.lineCap = "round";
  context.lineJoin = "round";
  document.constraints.forEach((constraint) => {
    if (constraint.type !== "perpendicular") return;
    const first = bodyEnds(constraint.objects[0]);
    const second = bodyEnds(constraint.objects[1]);
    if (!first || !second) return;

    const firstDirection = direction(first);
    const secondDirection = direction(second);
    const denominator = firstDirection.x * secondDirection.y
      - firstDirection.y * secondDirection.x;
    if (Math.abs(denominator) < 0.0001) return;
    const offsetX = second.start.x - first.start.x;
    const offsetY = second.start.y - first.start.y;
    const scale = (offsetX * secondDirection.y - offsetY * secondDirection.x) / denominator;
    const intersection = {
      x: first.start.x + firstDirection.x * scale,
      y: first.start.y + firstDirection.y * scale,
    };
    const inwardDirection = (
      line: { start: Body; end: Body },
      objectId: string,
      fallback: { x: number; y: number },
    ) => {
      if (objects.get(objectId)?.type !== "segment") return fallback;
      const startDistance = Math.hypot(
        line.start.x - intersection.x,
        line.start.y - intersection.y,
      );
      const endDistance = Math.hypot(
        line.end.x - intersection.x,
        line.end.y - intersection.y,
      );
      const target = startDistance >= endDistance ? line.start : line.end;
      const dx = target.x - intersection.x;
      const dy = target.y - intersection.y;
      const length = Math.max(0.0001, Math.hypot(dx, dy));
      return { x: dx / length, y: dy / length };
    };
    const firstMarkerDirection = inwardDirection(
      first,
      constraint.objects[0],
      firstDirection,
    );
    const secondMarkerDirection = inwardDirection(
      second,
      constraint.objects[1],
      secondDirection,
    );
    const size = 9;
    const firstCorner = {
      x: intersection.x + firstMarkerDirection.x * size,
      y: intersection.y + firstMarkerDirection.y * size,
    };
    const secondCorner = {
      x: intersection.x + secondMarkerDirection.x * size,
      y: intersection.y + secondMarkerDirection.y * size,
    };
    context.beginPath();
    context.moveTo(firstCorner.x, firstCorner.y);
    context.lineTo(
      firstCorner.x + secondMarkerDirection.x * size,
      firstCorner.y + secondMarkerDirection.y * size,
    );
    context.lineTo(secondCorner.x, secondCorner.y);
    context.stroke();
  });
  context.restore();
}

type EdgePath = {
  edge: GraphEdgeSpec;
  source: CanvasPoint;
  target: CanvasPoint;
  control?: { x: number; y: number };
};

function graphEdgePaths(
  graph: GraphSpec,
  bodies: Map<string, Body>,
  edgeRepulsion: number,
): EdgePath[] {
  const totals = new Map<string, number>();
  graph.edges.forEach((edge) => {
    const key = [edge.source, edge.target].sort().join("::");
    totals.set(key, (totals.get(key) ?? 0) + 1);
  });
  const seen = new Map<string, number>();

  return graph.edges.flatMap((edge) => {
    const source = bodies.get(edge.source);
    const target = bodies.get(edge.target);
    if (!source || !target) return [];
    const key = [edge.source, edge.target].sort().join("::");
    const index = seen.get(key) ?? 0;
    seen.set(key, index + 1);
    const count = totals.get(key) ?? 1;
    if (count === 1 || edgeRepulsion === 0) return [{ edge, source, target }];

    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const length = Math.max(0.0001, Math.hypot(dx, dy));
    const canonicalDirection = edge.source < edge.target ? 1 : -1;
    const bend = (index - (count - 1) / 2) * edgeRepulsion * canonicalDirection;
    return [{
      edge,
      source,
      target,
      control: {
        x: (source.x + target.x) / 2 - dy / length * bend,
        y: (source.y + target.y) / 2 + dx / length * bend,
      },
    }];
  });
}

function drawGraphObjects(
  context: CanvasRenderingContext2D,
  graphs: GraphSpec[],
  bodies: Map<string, Body>,
  edgeRepulsion: number,
) {
  const renderedPaths: EdgePath[] = [];
  graphs.forEach((graph) => {
    const paths = graphEdgePaths(graph, bodies, edgeRepulsion);
    renderedPaths.push(...paths);

    context.save();
    context.strokeStyle = "#171814";
    context.lineWidth = 1.8;
    context.lineCap = "round";
    paths.forEach((path) => {
      context.setLineDash(path.edge.length === undefined ? [] : [7, 6]);
      context.beginPath();
      context.moveTo(path.source.x, path.source.y);
      if (path.control) {
        context.quadraticCurveTo(
          path.control.x,
          path.control.y,
          path.target.x,
          path.target.y,
        );
      } else {
        context.lineTo(path.target.x, path.target.y);
      }
      context.stroke();
    });
    context.restore();

  });
  return renderedPaths;
}

type LabelPlacement = {
  x: number;
  y: number;
  halfWidth: number;
  halfHeight: number;
};

function pointToSegmentDistance(
  x: number,
  y: number,
  start: { x: number; y: number },
  end: { x: number; y: number },
) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const denominator = dx * dx + dy * dy;
  const t = denominator === 0
    ? 0
    : Math.max(0, Math.min(1, ((x - start.x) * dx + (y - start.y) * dy) / denominator));
  return Math.hypot(x - (start.x + dx * t), y - (start.y + dy * t));
}

function pointToEdgeDistance(x: number, y: number, path: EdgePath) {
  if (!path.control) return pointToSegmentDistance(x, y, path.source, path.target);
  let minimum = Number.POSITIVE_INFINITY;
  let previous = { x: path.source.x, y: path.source.y };
  for (let step = 1; step <= 12; step += 1) {
    const t = step / 12;
    const inverse = 1 - t;
    const current = {
      x: inverse * inverse * path.source.x
        + 2 * inverse * t * path.control.x
        + t * t * path.target.x,
      y: inverse * inverse * path.source.y
        + 2 * inverse * t * path.control.y
        + t * t * path.target.y,
    };
    minimum = Math.min(minimum, pointToSegmentDistance(x, y, previous, current));
    previous = current;
  }
  return minimum;
}

function hoveredStructure(
  document: GeometryDocument,
  bodies: Map<string, Body>,
  pointer: CanvasPoint,
  width: number,
  height: number,
  edgeRepulsion: number,
  extensionLength: number,
) {
  const nearPoint = (ids: string[]) => ids.some((id) => {
    const body = bodies.get(id);
    return body && Math.hypot(pointer.x - body.x, pointer.y - body.y) <= 10;
  });
  const nearPath = (paths: EdgePath[]) =>
    paths.some((path) => pointToEdgeDistance(pointer.x, pointer.y, path) <= 7);

  return [...document.objects].reverse().find((object) => {
    if (!objectName(object)) return false;
    if (nearPoint(objectPointIds(object))) return true;

    if (object.type === "circle") {
      const center = bodies.get(object.center);
      return Boolean(
        center && Math.hypot(pointer.x - center.x, pointer.y - center.y) <= object.radius + 7,
      );
    }
    if (object.type === "graph") {
      return nearPath(graphEdgePaths(object, bodies, edgeRepulsion));
    }
    if (object.type === "line") {
      const visual = lineVisualEndpoints(object, document, bodies);
      if (!visual) return false;
      const { renderedStart, renderedEnd } = openLineGeometry(
        visual.start,
        visual.end,
        true,
        width,
        height,
        extensionLength,
      );
      return pointToSegmentDistance(
        pointer.x,
        pointer.y,
        renderedStart,
        renderedEnd,
      ) <= 7;
    }
    if (object.type === "ray") {
      const start = bodies.get(object.from);
      const through = bodies.get(object.through);
      if (!start || !through) return false;
      const { renderedStart, renderedEnd } = openLineGeometry(
        start,
        through,
        false,
        width,
        height,
        extensionLength,
      );
      return pointToSegmentDistance(
        pointer.x,
        pointer.y,
        renderedStart,
        renderedEnd,
      ) <= 7;
    }

    const paths: EdgePath[] = [];
    const add = (sourceId: string, targetId: string) => {
      const source = bodies.get(sourceId);
      const target = bodies.get(targetId);
      if (source && target) {
        paths.push({ edge: { source: sourceId, target: targetId }, source, target });
      }
    };
    if (object.type === "segment") add(...object.between);
    if (object.type === "angle") {
      add(object.arms[1], object.arms[0]);
      add(object.arms[1], object.arms[2]);
    }
    if (object.type === "polygon") {
      object.vertices.forEach((vertex, index) => {
        add(vertex, object.vertices[(index + 1) % object.vertices.length]);
      });
    }
    return nearPath(paths);
  })?.id;
}

function pointToBoxDistance(
  x: number,
  y: number,
  box: LabelPlacement,
) {
  const dx = Math.max(0, Math.abs(x - box.x) - box.halfWidth);
  const dy = Math.max(0, Math.abs(y - box.y) - box.halfHeight);
  return Math.hypot(dx, dy);
}

function edgeToBoxDistance(path: EdgePath, box: LabelPlacement) {
  let minimum = Number.POSITIVE_INFINITY;
  const steps = path.control ? 16 : 12;
  for (let step = 0; step <= steps; step += 1) {
    const t = step / steps;
    const inverse = 1 - t;
    const x = path.control
      ? inverse * inverse * path.source.x
        + 2 * inverse * t * path.control.x
        + t * t * path.target.x
      : path.source.x + (path.target.x - path.source.x) * t;
    const y = path.control
      ? inverse * inverse * path.source.y
        + 2 * inverse * t * path.control.y
        + t * t * path.target.y
      : path.source.y + (path.target.y - path.source.y) * t;
    minimum = Math.min(minimum, pointToBoxDistance(x, y, box));
  }
  return minimum;
}

function layoutStructureLabels(
  context: CanvasRenderingContext2D,
  document: GeometryDocument,
  bodies: Map<string, Body>,
  paths: EdgePath[],
  canvasWidth: number,
  canvasHeight: number,
  visibleObjectId?: string,
) {
  context.save();
  context.font = "600 10px 'DM Mono', monospace";
  const placements = new Map<string, LabelPlacement>();
  const structures: Array<{
    id: string;
    name: string;
    candidates: (halfWidth: number) => Array<{ x: number; y: number }>;
  }> = [];

  document.objects.forEach((object) => {
    if (object.id !== visibleObjectId || object.type !== "circle" || !object.name) return;
    const center = bodies.get(object.center);
    if (!center) return;
    structures.push({
      id: object.id,
      name: object.name,
      candidates: (halfWidth) => [
        { x: center.x, y: center.y - object.radius - 16 },
        { x: center.x, y: center.y + object.radius + 16 },
        { x: center.x - object.radius - halfWidth - 12, y: center.y },
        { x: center.x + object.radius + halfWidth + 12, y: center.y },
      ],
    });
  });

  document.objects.forEach((object) => {
    const name = objectName(object);
    if (object.id !== visibleObjectId || object.type === "circle" || !name) return;
    const nodes = objectPointIds(object)
      .map((id) => bodies.get(id))
      .filter((body): body is Body => Boolean(body));
    if (!nodes.length) return;
    const minX = Math.min(...nodes.map((body) => body.x));
    const maxX = Math.max(...nodes.map((body) => body.x));
    const minY = Math.min(...nodes.map((body) => body.y));
    const maxY = Math.max(...nodes.map((body) => body.y));
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    structures.push({
      id: object.id,
      name,
      candidates: (halfWidth) => [
        { x: centerX, y: minY - 27 },
        { x: centerX, y: maxY + 27 },
        { x: minX - halfWidth - 16, y: centerY },
        { x: maxX + halfWidth + 16, y: centerY },
      ],
    });
  });

  structures.forEach((structure) => {
    const width = Math.max(6, context.measureText(structure.name).width);
    const halfWidth = width / 2;
    const halfHeight = 5;
    let best: (LabelPlacement & { score: number }) | undefined;
    structure.candidates(halfWidth).forEach((candidate, candidateIndex) => {
      const placement = { ...candidate, halfWidth, halfHeight };
      let score = candidateIndex * 12;
      if (placement.x - halfWidth < 8) score += (8 - placement.x + halfWidth) * 100;
      if (placement.x + halfWidth > canvasWidth - 8) {
        score += (placement.x + halfWidth - canvasWidth + 8) * 100;
      }
      if (placement.y - halfHeight < 8) score += (8 - placement.y + halfHeight) * 100;
      if (placement.y + halfHeight > canvasHeight - 8) {
        score += (placement.y + halfHeight - canvasHeight + 8) * 100;
      }

      bodies.forEach((body) => {
        const overlapX = halfWidth + 8 - Math.abs(placement.x - body.x);
        const overlapY = halfHeight + 8 - Math.abs(placement.y - body.y);
        if (overlapX > 0 && overlapY > 0) score += overlapX * overlapY * 28;
      });
      paths.forEach((path) => {
        const overlap = 4 - edgeToBoxDistance(path, placement);
        if (overlap > 0) score += overlap * overlap * 40;
      });
      placements.forEach((other) => {
        const overlapX = halfWidth + other.halfWidth + 6 - Math.abs(placement.x - other.x);
        const overlapY = halfHeight + other.halfHeight + 6 - Math.abs(placement.y - other.y);
        if (overlapX > 0 && overlapY > 0) score += overlapX * overlapY * 40;
      });

      if (!best || score < best.score) best = { ...placement, score };
    });
    if (best) placements.set(structure.id, best);
  });
  context.restore();
  return placements;
}

function drawStructureLabels(
  context: CanvasRenderingContext2D,
  document: GeometryDocument,
  placements: Map<string, LabelPlacement>,
) {
  context.save();
  context.fillStyle = "#171814";
  context.font = "600 10px 'DM Mono', monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  document.objects.forEach((object) => {
    const name = objectName(object);
    const placement = placements.get(object.id);
    if (name && placement) context.fillText(name, placement.x, placement.y);
  });
  context.restore();
}

function layoutCompactLabels(
  context: CanvasRenderingContext2D,
  document: GeometryDocument,
  bodies: Map<string, Body>,
  paths: EdgePath[],
  labels: Map<string, string>,
  reservedPlacements: LabelPlacement[],
) {
  const preferredDirections = new Map<string, { x: number; y: number }>();
  document.objects.forEach((object) => {
    if (object.type === "circle") return;
    const nodes = objectPointIds(object)
      .map((id) => bodies.get(id))
      .filter((body): body is Body => Boolean(body));
    if (!nodes.length) return;
    const centerX = nodes.reduce((sum, body) => sum + body.x, 0) / nodes.length;
    const centerY = nodes.reduce((sum, body) => sum + body.y, 0) / nodes.length;
    nodes.forEach((body) => {
      const dx = body.x - centerX;
      const dy = body.y - centerY;
      const length = Math.hypot(dx, dy);
      preferredDirections.set(
        body.id,
        length < 0.001 ? { x: 0, y: -1 } : { x: dx / length, y: dy / length },
      );
    });
  });

  const objects = new Map(document.objects.map((object) => [object.id, object]));
  document.constraints.forEach((constraint) => {
    if (constraint.type !== "on") return;
    const object = objects.get(constraint.object);
    if (object?.type !== "circle") return;
    const point = bodies.get(constraint.point);
    const center = bodies.get(object.center);
    if (!point || !center) return;
    const dx = point.x - center.x;
    const dy = point.y - center.y;
    const length = Math.hypot(dx, dy);
    if (length > 0.001) {
      preferredDirections.set(point.id, { x: dx / length, y: dy / length });
    }
  });

  const constraintSegments = document.constraints.flatMap((constraint) => {
    if (constraint.type === "distance") {
      const start = bodies.get(constraint.a);
      const end = bodies.get(constraint.b);
      return start && end ? [{ start, end }] : [];
    }
    if (constraint.type !== "on") return [];
    const object = objects.get(constraint.object);
    if (object?.type !== "circle") return [];
    const start = bodies.get(object.center);
    const end = bodies.get(constraint.point);
    return start && end ? [{ start, end }] : [];
  });

  context.save();
  context.font = "600 10px 'DM Mono', monospace";
  const placements = new Map<string, LabelPlacement>();
  const occupied = [...reservedPlacements];
  document.points.forEach((point) => {
    const body = bodies.get(point.id);
    if (!body) return;
    const label = labels.get(point.id) ?? point.id;
    const width = Math.max(6, context.measureText(label).width);
    const halfWidth = width / 2;
    const halfHeight = 5;
    const radius = Math.hypot(halfWidth, halfHeight);
    const preferred = preferredDirections.get(point.id) ?? { x: 0, y: -1 };
    const candidateDirections = [
      preferred,
      ...Array.from({ length: 16 }, (_, index) => {
        const angle = index / 16 * Math.PI * 2;
        return { x: Math.cos(angle), y: Math.sin(angle) };
      }),
    ];
    let best = {
      x: body.x,
      y: body.y - 14,
      halfWidth,
      halfHeight,
      score: Number.POSITIVE_INFINITY,
    };

    candidateDirections.forEach((direction) => {
      const anchorDistance = 11 + Math.min(7, width / 2);
      const x = body.x + direction.x * anchorDistance;
      const y = body.y + direction.y * anchorDistance;
      let score = (1 - (direction.x * preferred.x + direction.y * preferred.y)) * 18;

      bodies.forEach((other) => {
        const overlapX = halfWidth + 7 - Math.abs(x - other.x);
        const overlapY = halfHeight + 7 - Math.abs(y - other.y);
        if (overlapX > 0 && overlapY > 0) score += overlapX * overlapY * 24;
      });
      paths.forEach((path) => {
        const overlap = radius + 3 - pointToEdgeDistance(x, y, path);
        if (overlap > 0) score += overlap * overlap * 18;
      });
      constraintSegments.forEach(({ start, end }) => {
        const overlap = radius + 3 - pointToSegmentDistance(x, y, start, end);
        if (overlap > 0) score += overlap * overlap * 18;
      });
      occupied.forEach((placement) => {
        const overlapX = halfWidth + placement.halfWidth + 3 - Math.abs(x - placement.x);
        const overlapY = halfHeight + placement.halfHeight + 3 - Math.abs(y - placement.y);
        if (overlapX > 0 && overlapY > 0) score += overlapX * overlapY * 30;
      });

      if (score < best.score) best = { x, y, halfWidth, halfHeight, score };
    });
    placements.set(point.id, best);
    occupied.push(best);
  });
  context.restore();
  return placements;
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

function drawPoint(
  context: CanvasRenderingContext2D,
  body: Body,
  label: string,
  compact: boolean,
  labelPlacement?: LabelPlacement,
) {
  if (compact) {
    context.save();
    context.fillStyle = "#171814";
    context.beginPath();
    context.arc(body.x, body.y, 4, 0, Math.PI * 2);
    context.fill();
    context.font = "600 10px 'DM Mono', monospace";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(
      label,
      labelPlacement?.x ?? body.x,
      labelPlacement?.y ?? body.y - 14,
    );
    context.restore();
    return;
  }

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
    } else if (constraint.type === "on") {
      const object = objects.get(constraint.object);
      if (object?.type !== "circle") return;
      a = bodies.get(object.center);
      b = bodies.get(constraint.point);
    } else {
      return;
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

function InfoTip({ label, children }: { label: string; children: string }) {
  const tooltipId = useId();
  return (
    <span className="info-tip">
      <button
        type="button"
        className="info-tip-button"
        aria-label={`About ${label}`}
        aria-describedby={tooltipId}
      >
        ?
      </button>
      <span id={tooltipId} className="info-tooltip" role="tooltip">
        {children}
      </span>
    </span>
  );
}

function SidebarSection({
  title,
  open,
  onToggle,
  action,
  children,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  action?: ReactNode;
  children: ReactNode;
}) {
  const contentId = useId();
  return (
    <section className={`sidebar-section${open ? " is-open" : " is-collapsed"}`}>
      <div className="section-heading">
        <button
          type="button"
          className="section-toggle"
          aria-expanded={open}
          aria-controls={contentId}
          onClick={onToggle}
        >
          <span>{title}</span>
        </button>
        {open ? action : null}
      </div>
      {open ? (
        <div id={contentId} className="section-content">
          {children}
        </div>
      ) : null}
    </section>
  );
}

function RangeControl({
  label,
  help,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  help: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="physics-control">
      <div className="control-row">
        <span className="control-label">
          {label}
          <InfoTip label={label}>{help}</InfoTip>
        </span>
        <output>{display}</output>
      </div>
      <input
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onInput={(event) => onChange(Number(event.currentTarget.value))}
      />
    </div>
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
    screenWidth: 0,
    screenHeight: 0,
  });
  const dragRef = useRef<DragState | null>(null);
  const hoverPointerRef = useRef<CanvasPoint | null>(null);
  const [document, setDocument] = useState(allStructuresDocument);
  const [draft, setDraft] = useState(() => formatDocument(allStructuresDocument));
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const [resetToken, setResetToken] = useState(0);
  const [physics, setPhysics] = useState(defaultPhysics);
  const [showVectors, setShowVectors] = useState(false);
  const [compactPoints, setCompactPoints] = useState(true);
  const [codeOpen, setCodeOpen] = useState(true);
  const [viewOpen, setViewOpen] = useState(true);
  const [simulationOpen, setSimulationOpen] = useState(false);
  const [zoomPercent, setZoomPercent] = useState(100);

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
    const zoomScale = zoomPercent / 100;
    const visualDocument: GeometryDocument = {
      ...document,
      objects: document.objects.map((object) =>
        object.type === "circle"
          ? { ...object, radius: object.radius * zoomScale }
          : object),
    };
    let animationFrame = 0;
    let previousTime = performance.now();

    const resize = () => {
      const bounds = panel.getBoundingClientRect();
      const screenWidth = Math.max(320, bounds.width);
      const screenHeight = Math.max(420, bounds.height);
      const width = screenWidth / zoomScale;
      const height = screenHeight / zoomScale;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(screenWidth * dpr);
      canvas.height = Math.round(screenHeight * dpr);
      canvas.style.width = `${screenWidth}px`;
      canvas.style.height = `${screenHeight}px`;
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
      world.screenWidth = screenWidth;
      world.screenHeight = screenHeight;
    };

    const observer = new ResizeObserver(resize);
    observer.observe(panel);
    resize();

    const draw = () => {
      const screenBodies = new Map(
        [...world.bodies].map(([id, body]) => [id, {
          ...body,
          x: body.x * zoomScale,
          y: body.y * zoomScale,
        }]),
      );
      context.clearRect(0, 0, world.screenWidth, world.screenHeight);
      context.fillStyle = "#f7f6ef";
      context.fillRect(0, 0, world.screenWidth, world.screenHeight);
      drawGrid(context, world.screenWidth, world.screenHeight);

      visualDocument.objects.forEach((object) => {
        if (object.type !== "circle") return;
        drawImplicitContour(
          context,
          world.screenWidth,
          world.screenHeight,
          (x, y) => objectDistance(object, x, y, screenBodies),
          "#171814",
          2,
        );
      });
      drawPrimitiveObjects(
        context,
        visualDocument,
        screenBodies,
        world.screenWidth,
        world.screenHeight,
        physics.extensionLength,
      );
      drawRelationMarkers(context, visualDocument, screenBodies);
      const renderedGraphPaths = drawGraphObjects(
        context,
        graphObjects,
        screenBodies,
        physics.edgeRepulsion,
      );
      const renderedEdgePaths = [
        ...renderedGraphPaths,
        ...primitiveEdgePaths(
          visualDocument,
          screenBodies,
          world.screenWidth,
          world.screenHeight,
          physics.extensionLength,
        ),
      ];
      const hoveredObjectId = hoverPointerRef.current
        ? hoveredStructure(
            visualDocument,
            screenBodies,
            hoverPointerRef.current,
            world.screenWidth,
            world.screenHeight,
            physics.edgeRepulsion,
            physics.extensionLength,
          )
        : undefined;
      const structureLabelPlacements = layoutStructureLabels(
        context,
        visualDocument,
        screenBodies,
        renderedEdgePaths,
        world.screenWidth,
        world.screenHeight,
        hoveredObjectId,
      );
      const compactLabelPlacements = compactPoints
        ? layoutCompactLabels(
            context,
            visualDocument,
            screenBodies,
            renderedEdgePaths,
            pointLabels,
            [...structureLabelPlacements.values()],
          )
        : undefined;
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
          screenBodies,
          forceBreakdowns,
          world.screenWidth,
        );
      }
      drawConstraintLinks(context, visualDocument, screenBodies);
      drawStructureLabels(context, visualDocument, structureLabelPlacements);
      screenBodies.forEach((body) => drawPoint(
        context,
        body,
        pointLabels.get(body.id) ?? body.id,
        compactPoints,
        compactLabelPlacements?.get(body.id),
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
    physics.edgeRepulsion,
    physics.extensionLength,
    physics.damping,
    pointLabels,
    showVectors,
    compactPoints,
    zoomPercent,
    resetToken,
  ]);

  const screenPointerPosition = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  };

  const pointerPosition = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const screen = screenPointerPosition(event);
    const scale = zoomPercent / 100;
    return {
      x: screen.x / scale,
      y: screen.y / scale,
    };
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    hoverPointerRef.current = screenPointerPosition(event);
    const point = pointerPosition(event);
    let nearest: Body | undefined;
    let nearestDistance = 30 / (zoomPercent / 100);
    worldRef.current.bodies.forEach((body) => {
      const distance = Math.hypot(point.x - body.x, point.y - body.y);
      if (distance <= nearestDistance) {
        nearest = body;
        nearestDistance = distance;
      }
    });
    if (!nearest) return;
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
    hoverPointerRef.current = screenPointerPosition(event);
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
        const margin = 26 / (zoomPercent / 100);
        body.x = Math.max(margin, Math.min(worldRef.current.width - margin, body.x));
        body.y = Math.max(margin, Math.min(worldRef.current.height - margin, body.y));
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

        <SidebarSection
          title="Code"
          open={codeOpen}
          onToggle={() => setCodeOpen((open) => !open)}
          action={(
            <button type="button" className="text-button" onClick={formatDraft}>Format</button>
          )}
        >
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
        </SidebarSection>

        <SidebarSection
          title="View"
          open={viewOpen}
          onToggle={() => setViewOpen((open) => !open)}
        >
          <div className="physics-controls">
            <RangeControl
              label="Zoom"
              help="Scales the complete diagram around the center of the canvas without changing its geometry."
              value={zoomPercent}
              min={25}
              max={200}
              step={5}
              display={`${zoomPercent}%`}
              onChange={setZoomPercent}
            />
          </div>
        </SidebarSection>

        <SidebarSection
          title="Simulation"
          open={simulationOpen}
          onToggle={() => setSimulationOpen((open) => !open)}
          action={(
            <button
              type="button"
              className="text-button"
              onClick={() => setPhysics(defaultPhysics)}
            >
              Defaults
            </button>
          )}
        >
          <div className="physics-controls">
            <RangeControl
              label="Edge pull"
              help="How strongly connected nodes pull toward one another. Higher values tighten graphs faster."
              value={physics.attraction}
              min={0}
              max={80}
              step={1}
              display={String(physics.attraction)}
              onChange={(attraction) => setPhysics((current) => ({ ...current, attraction }))}
            />
            <RangeControl
              label="Node repulsion"
              help="How strongly nodes push apart. Higher values spread structures farther open."
              value={physics.repulsion}
              min={0}
              max={500}
              step={5}
              display={String(physics.repulsion)}
              onChange={(repulsion) => setPhysics((current) => ({ ...current, repulsion }))}
            />
            <RangeControl
              label="Edge repulsion"
              help="How far duplicate edges bow away from one another. At zero, duplicate edges overlap."
              value={physics.edgeRepulsion}
              min={0}
              max={96}
              step={1}
              display={String(physics.edgeRepulsion)}
              onChange={(edgeRepulsion) =>
                setPhysics((current) => ({ ...current, edgeRepulsion }))}
            />
            <RangeControl
              label="Open extension"
              help="How far lines and rays continue while fading beyond their defining points or constrained intersections."
              value={physics.extensionLength}
              min={16}
              max={180}
              step={4}
              display={`${physics.extensionLength}px`}
              onChange={(extensionLength) =>
                setPhysics((current) => ({ ...current, extensionLength }))}
            />
            <RangeControl
              label="Damping"
              help="How quickly motion loses energy. Higher values reduce drifting and settle sooner."
              value={physics.damping}
              min={0.5}
              max={14}
              step={0.1}
              display={physics.damping.toFixed(1)}
              onChange={(damping) => setPhysics((current) => ({ ...current, damping }))}
            />
            <div className="switch-toggle">
              <span className="control-label">
                Force vectors
                <InfoTip label="Force vectors">
                  Shows attraction, repulsion, and net-force arrows for every point.
                </InfoTip>
              </span>
              <input
                aria-label="Force vectors"
                type="checkbox"
                checked={showVectors}
                onChange={(event) => setShowVectors(event.currentTarget.checked)}
              />
            </div>
            <div className="switch-toggle">
              <span className="control-label">
                Compact points
                <InfoTip label="Compact points">
                  Switches between small dots with floating labels and large labeled circles.
                </InfoTip>
              </span>
              <input
                aria-label="Compact points"
                type="checkbox"
                checked={compactPoints}
                onChange={(event) => setCompactPoints(event.currentTarget.checked)}
              />
            </div>
          </div>
          <div className="button-row">
            <button type="button" onClick={() => setPaused((current) => !current)}>
              {paused ? "Resume" : "Pause"}
            </button>
            <button type="button" onClick={reset}>Reset</button>
          </div>
        </SidebarSection>
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
          onPointerLeave={() => {
            hoverPointerRef.current = null;
          }}
        />
      </section>
    </main>
  );
}
