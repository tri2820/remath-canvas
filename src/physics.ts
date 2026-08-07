export type Vector = {
  x: number;
  y: number;
};

export type Body = {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  fixed: boolean;
};

export type PhysicsConstraint = {
  particleIds: string[];
  error: (bodies: Map<string, Body>) => number;
  gradient?: (bodies: Map<string, Body>, particleId: string) => Vector;
};

export type PointerTarget = {
  particleId: string;
  x: number;
  y: number;
};

export type PositionSnapshot = Map<string, Vector>;

const NUMERIC_GRADIENT_STEP = 0.01;

function numericGradient(
  constraint: PhysicsConstraint,
  bodies: Map<string, Body>,
  particleId: string,
): Vector {
  const body = bodies.get(particleId);
  if (!body) return { x: 0, y: 0 };

  body.x += NUMERIC_GRADIENT_STEP;
  const errorRight = constraint.error(bodies);
  body.x -= NUMERIC_GRADIENT_STEP * 2;
  const errorLeft = constraint.error(bodies);
  body.x += NUMERIC_GRADIENT_STEP;

  body.y += NUMERIC_GRADIENT_STEP;
  const errorDown = constraint.error(bodies);
  body.y -= NUMERIC_GRADIENT_STEP * 2;
  const errorUp = constraint.error(bodies);
  body.y += NUMERIC_GRADIENT_STEP;

  return {
    x: (errorRight - errorLeft) / (NUMERIC_GRADIENT_STEP * 2),
    y: (errorDown - errorUp) / (NUMERIC_GRADIENT_STEP * 2),
  };
}

export function solveConstraints(
  constraints: PhysicsConstraint[],
  bodies: Map<string, Body>,
  draggedId: string | null,
  iterations = 8,
) {
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    constraints.forEach((constraint) => {
      const error = constraint.error(bodies);
      if (!Number.isFinite(error) || Math.abs(error) < 0.00001) return;

      const particles = constraint.particleIds
        .map((id) => bodies.get(id))
        .filter((body): body is Body => Boolean(body));
      const draggedHasMovablePartner = draggedId !== null && particles.some(
        (body) => !body.fixed && body.id !== draggedId,
      );
      const correctionWeight = (body: Body) =>
        body.id === draggedId && draggedHasMovablePartner ? 0 : 1;
      const gradients = new Map<string, Vector>();
      let denominator = 0;

      particles.forEach((body) => {
        if (body.fixed) return;
        const gradient = constraint.gradient
          ? constraint.gradient(bodies, body.id)
          : numericGradient(constraint, bodies, body.id);
        gradients.set(body.id, gradient);
        const weight = correctionWeight(body);
        denominator += weight * (gradient.x * gradient.x + gradient.y * gradient.y);
      });

      if (denominator < 0.000001) return;
      const scale = error / denominator;
      particles.forEach((body) => {
        if (body.fixed) return;
        const gradient = gradients.get(body.id);
        if (!gradient) return;
        const weight = correctionWeight(body);
        body.x -= gradient.x * scale * weight;
        body.y -= gradient.y * scale * weight;
      });
    });
  }
}

export function connectedComponents(
  constraints: PhysicsConstraint[],
  bodies: Map<string, Body>,
) {
  const neighbors = new Map<string, Set<string>>();
  bodies.forEach((_, id) => neighbors.set(id, new Set()));
  constraints.forEach((constraint) => {
    constraint.particleIds.forEach((left) => {
      constraint.particleIds.forEach((right) => {
        if (left !== right) neighbors.get(left)?.add(right);
      });
    });
  });

  const remaining = new Set(bodies.keys());
  const components: Set<string>[] = [];
  while (remaining.size) {
    const first = remaining.values().next().value as string;
    const component = new Set<string>([first]);
    const pending = [first];
    while (pending.length) {
      const current = pending.pop()!;
      neighbors.get(current)?.forEach((neighbor) => {
        if (component.has(neighbor)) return;
        component.add(neighbor);
        pending.push(neighbor);
      });
    }
    component.forEach((id) => remaining.delete(id));
    components.push(component);
  }
  return components;
}

export function integrate(
  bodies: Map<string, Body>,
  deltaSeconds: number,
  pointerTarget: PointerTarget | null,
): PositionSnapshot {
  const previous = new Map<string, Vector>();
  bodies.forEach((body) => {
    previous.set(body.id, { x: body.x, y: body.y });
    if (body.fixed) {
      body.vx = 0;
      body.vy = 0;
      return;
    }

    if (pointerTarget?.particleId === body.id) {
      body.x = pointerTarget.x;
      body.y = pointerTarget.y;
      return;
    }

    const freeDecay = Math.exp(-2.4 * deltaSeconds);
    body.vx *= freeDecay;
    body.vy *= freeDecay;

    const speed = Math.hypot(body.vx, body.vy);
    if (speed > 1800) {
      body.vx = body.vx / speed * 1800;
      body.vy = body.vy / speed * 1800;
    }
    body.x += body.vx * deltaSeconds;
    body.y += body.vy * deltaSeconds;
  });
  return previous;
}

export function reconcileVelocities(
  bodies: Map<string, Body>,
  previous: PositionSnapshot,
  deltaSeconds: number,
  boundaryCorrections: Map<string, Vector>,
) {
  if (deltaSeconds <= 0) return;
  bodies.forEach((body) => {
    if (body.fixed) return;
    const oldPosition = previous.get(body.id);
    if (!oldPosition) return;
    body.vx = (body.x - oldPosition.x) / deltaSeconds;
    body.vy = (body.y - oldPosition.y) / deltaSeconds;

    const speed = Math.hypot(body.vx, body.vy);
    if (speed > 1800) {
      body.vx = body.vx / speed * 1800;
      body.vy = body.vy / speed * 1800;
    }

    const correction = boundaryCorrections.get(body.id);
    if (correction?.x && Math.sign(body.vx) !== Math.sign(correction.x)) body.vx = 0;
    if (correction?.y && Math.sign(body.vy) !== Math.sign(correction.y)) body.vy = 0;
  });
}
