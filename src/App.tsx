import { useEffect, useMemo, useRef, useState } from "react";
import {
  type Body,
  type PhysicsConstraint,
  connectedComponents,
  integrate,
  reconcileVelocities,
  solveConstraints,
} from "./physics";

type PointSpec = {
  id: string;
  fixed?: boolean;
};

type CircleSpec = {
  id: string;
  type: "circle";
  center: string;
  radius: number;
};

type LineSpec = {
  id: string;
  type: "line";
  through: [string, string];
};

type ObjectSpec = CircleSpec | LineSpec;

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

const initialDocument: GeometryDocument = {
  points: [
    { id: "O" },
    { id: "P" },
  ],
  objects: [
    { id: "c", type: "circle", center: "O", radius: 180 },
  ],
  constraints: [
    { type: "on", point: "P", object: "c" },
  ],
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
      return { id: object.id, type: "circle", center: object.center, radius: object.radius };
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

    throw new Error(`Object ${index + 1} has an unsupported type.`);
  });

  const objectIds = new Set(objects.map((object) => object.id));
  if (objectIds.size !== objects.length) throw new Error("Object ids must be unique.");

  const constraints = candidate.constraints.map((constraint, index): ConstraintSpec => {
    if (!constraint || typeof constraint !== "object") {
      throw new Error(`Constraint ${index + 1} must be an object.`);
    }

    if (constraint.type === "on") {
      if (
        typeof constraint.point !== "string" ||
        !pointIds.has(constraint.point) ||
        typeof constraint.object !== "string" ||
        !objectIds.has(constraint.object)
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
  const layoutRadius = Math.min(world.width, world.height) * 0.18;
  document.points.forEach((point, index) => {
    const angle = (index / Math.max(1, document.points.length)) * Math.PI * 2;
    const offset = document.points.length === 1 ? 0 : layoutRadius;
    bodies.set(point.id, {
      id: point.id,
      x: world.width / 2 + Math.cos(angle) * offset,
      y: world.height / 2 + Math.sin(angle) * offset,
      vx: 0,
      vy: 0,
      fixed: point.fixed === true,
    });
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

  const a = bodies.get(object.through[0]);
  const b = bodies.get(object.through[1]);
  if (!a || !b) return 1;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.max(0.0001, Math.hypot(dx, dy));
  return ((x - a.x) * dy - (y - a.y) * dx) / length;
}

function distanceEquation(aId: string, bId: string, target: number): PhysicsConstraint {
  return {
    particleIds: [aId, bId],
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
  return document.constraints.flatMap((constraint) => {
    if (constraint.type === "distance") {
      return [distanceEquation(constraint.a, constraint.b, constraint.value)];
    }

    const object = objects.get(constraint.object);
    if (!object) return [];
    if (object.type === "circle") {
      return [distanceEquation(object.center, constraint.point, object.radius)];
    }

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
}

function componentBounds(
  ids: Set<string>,
  document: GeometryDocument,
  bodies: Map<string, Body>,
) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  ids.forEach((id) => {
    const body = bodies.get(id);
    if (!body) return;
    minX = Math.min(minX, body.x - 22);
    minY = Math.min(minY, body.y - 22);
    maxX = Math.max(maxX, body.x + 22);
    maxY = Math.max(maxY, body.y + 22);
  });

  document.objects.forEach((object) => {
    if (object.type !== "circle" || !ids.has(object.center)) return;
    const center = bodies.get(object.center);
    if (!center) return;
    minX = Math.min(minX, center.x - object.radius);
    minY = Math.min(minY, center.y - object.radius);
    maxX = Math.max(maxX, center.x + object.radius);
    maxY = Math.max(maxY, center.y + object.radius);
  });

  return { minX, minY, maxX, maxY };
}

function containComponent(
  ids: Set<string>,
  document: GeometryDocument,
  bodies: Map<string, Body>,
  width: number,
  height: number,
) {
  const margin = 18;
  const bounds = componentBounds(ids, document, bodies);
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

  if (!correctionX && !correctionY) return { x: 0, y: 0 };
  ids.forEach((id) => {
    const body = bodies.get(id);
    if (!body || body.fixed) return;
    body.x += correctionX;
    body.y += correctionY;
  });
  return { x: correctionX, y: correctionY };
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

function drawPoint(context: CanvasRenderingContext2D, body: Body) {
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
  context.fillText(body.id, body.x, body.y + 0.5);
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
  context.setLineDash([6, 6]);
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

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasPanelRef = useRef<HTMLDivElement>(null);
  const worldRef = useRef<World>({ bodies: new Map(), initialized: false, width: 0, height: 0 });
  const dragRef = useRef<DragState | null>(null);
  const [document, setDocument] = useState(initialDocument);
  const [draft, setDraft] = useState(formatDocument(initialDocument));
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const [resetToken, setResetToken] = useState(0);

  const summary = useMemo(
    () => `${document.points.length} points · ${document.objects.length} objects`,
    [document],
  );
  const physicsConstraints = useMemo(
    () => compilePhysicsConstraints(document),
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
        drawImplicitContour(
          context,
          world.width,
          world.height,
          (x, y) => objectDistance(object, x, y, world.bodies),
        );
      });
      drawConstraintLinks(context, document, world.bodies);
      world.bodies.forEach((body) => drawPoint(context, body));
    };

    const tick = (time: number) => {
      const deltaSeconds = Math.min(1 / 30, Math.max(0, (time - previousTime) / 1000));
      previousTime = time;

      if (!paused) {
        const drag = dragRef.current;
        const components = connectedComponents(physicsConstraints, world.bodies);
        const pointerTarget = drag
          ? {
              particleId: drag.id,
              x: drag.x + drag.offsetX,
              y: drag.y + drag.offsetY,
            }
          : null;
        const previousPositions = integrate(world.bodies, deltaSeconds, pointerTarget);

        solveConstraints(physicsConstraints, world.bodies, drag?.id ?? null);

        const boundaryCorrections = new Map<string, { x: number; y: number }>();
        components.forEach((component) => {
          const correction = containComponent(
            component,
            document,
            world.bodies,
            world.width,
            world.height,
          );
          component.forEach((id) => boundaryCorrections.set(id, correction));
        });

        reconcileVelocities(
          world.bodies,
          previousPositions,
          deltaSeconds,
          boundaryCorrections,
        );
      }
      draw();
      animationFrame = requestAnimationFrame(tick);
    };

    animationFrame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
    };
  }, [document, paused, physicsConstraints, resetToken]);

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
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const reset = () => {
    worldRef.current.initialized = false;
    setResetToken((token) => token + 1);
  };

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <header>
          <h1>ReMath</h1>
          <span>{summary}</span>
        </header>

        <section className="code-section">
          <div className="section-heading">
            <h2>Code</h2>
            <button type="button" className="text-button" onClick={formatDraft}>Format</button>
          </div>
          <textarea
            aria-label="Geometry code"
            spellCheck={false}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
          />
          {error ? <p className="error-message">{error}</p> : null}
          <button type="button" className="apply-button" onClick={applyDraft}>Apply</button>
        </section>

        <section className="physics-section">
          <div className="section-heading"><h2>Simulation</h2></div>
          <div className="button-row">
            <button type="button" onClick={() => setPaused((current) => !current)}>
              {paused ? "Resume" : "Pause"}
            </button>
            <button type="button" onClick={reset}>Reset</button>
          </div>
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
        />
      </section>
    </main>
  );
}
