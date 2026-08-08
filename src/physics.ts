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
};

export type PhysicsConstraint = {
  particleIds: string[];
  error: (bodies: Map<string, Body>) => number;
  gradient?: (bodies: Map<string, Body>, particleId: string) => Vector;
  stiffness?: number;
};

export type PhysicsAttraction = {
  a: string;
  b: string;
  strength: number;
};

export type PhysicsRepulsor = {
  anchorId: string;
  charge: number;
  excludes?: string[];
};

export type PointerTarget = {
  particleId: string;
  x: number;
  y: number;
};

export type PositionSnapshot = Map<string, Vector>;

export type ForceBreakdown = {
  attraction: Vector;
  repulsion: Vector;
  net: Vector;
};

export type PhysicsParameters = {
  repulsionStrength: number;
  damping: number;
};

const NUMERIC_GRADIENT_STEP = 0.01;
const REPULSION_UNIT_SCALE = 100_000;
const REPULSION_SOFTENING_SQUARED = 20 * 20;
const EXTERNAL_STRUCTURE_REPULSION = 0.025;
const SETTLE_SPEED = 4;

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
      const draggedHasPartner = draggedId !== null && particles.some(
        (body) => body.id !== draggedId,
      );
      const correctionWeight = (body: Body) =>
        body.id === draggedId && draggedHasPartner ? 0 : 1;
      const gradients = new Map<string, Vector>();
      let denominator = 0;

      particles.forEach((body) => {
        const gradient = constraint.gradient
          ? constraint.gradient(bodies, body.id)
          : numericGradient(constraint, bodies, body.id);
        gradients.set(body.id, gradient);
        const weight = correctionWeight(body);
        denominator += weight * (gradient.x * gradient.x + gradient.y * gradient.y);
      });

      if (denominator < 0.000001) return;
      const stiffness = Math.max(0, Math.min(1, constraint.stiffness ?? 1));
      const scale = error / denominator * stiffness;
      particles.forEach((body) => {
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
  attractions: PhysicsAttraction[],
  bodies: Map<string, Body>,
) {
  const neighbors = new Map<string, Set<string>>();
  bodies.forEach((_, id) => neighbors.set(id, new Set()));
  const connect = (left: string, right: string) => {
    if (left === right) return;
    neighbors.get(left)?.add(right);
    neighbors.get(right)?.add(left);
  };
  constraints.forEach((constraint) => {
    constraint.particleIds.forEach((left) => {
      constraint.particleIds.forEach((right) => {
        connect(left, right);
      });
    });
  });
  attractions.forEach((attraction) => connect(attraction.a, attraction.b));

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

export function calculateForces(
  bodies: Map<string, Body>,
  attractions: PhysicsAttraction[],
  repulsors: PhysicsRepulsor[],
  components: Set<string>[],
  repulsionStrength: number,
): Map<string, ForceBreakdown> {
  const forces = new Map<string, ForceBreakdown>();
  bodies.forEach((_, id) => forces.set(id, {
    attraction: { x: 0, y: 0 },
    repulsion: { x: 0, y: 0 },
    net: { x: 0, y: 0 },
  }));
  const componentByParticle = new Map<string, number>();
  components.forEach((component, componentIndex) => {
    component.forEach((id) => componentByParticle.set(id, componentIndex));
  });
  const externalForces = components.map(() => ({ x: 0, y: 0 }));

  attractions.forEach((attraction) => {
    const a = bodies.get(attraction.a);
    const b = bodies.get(attraction.b);
    if (!a || !b) return;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const distance = Math.max(0.0001, Math.hypot(dx, dy));
    const force = distance * attraction.strength;
    const forceX = dx / distance * force;
    const forceY = dy / distance * force;
    const aForce = forces.get(a.id);
    const bForce = forces.get(b.id);
    if (aForce) {
      aForce.attraction.x += forceX;
      aForce.attraction.y += forceY;
    }
    if (bForce) {
      bForce.attraction.x -= forceX;
      bForce.attraction.y -= forceY;
    }
  });

  for (let left = 0; left < repulsors.length; left += 1) {
    for (let right = left + 1; right < repulsors.length; right += 1) {
      const leftRepulsor = repulsors[left];
      const rightRepulsor = repulsors[right];
      if (
        leftRepulsor.anchorId === rightRepulsor.anchorId ||
        leftRepulsor.excludes?.includes(rightRepulsor.anchorId) ||
        rightRepulsor.excludes?.includes(leftRepulsor.anchorId)
      ) {
        continue;
      }
      const a = bodies.get(leftRepulsor.anchorId);
      const b = bodies.get(rightRepulsor.anchorId);
      if (!a || !b) continue;
      const dx = b.x - a.x || 0.01;
      const dy = b.y - a.y || 0.01;
      const rawDistanceSquared = dx * dx + dy * dy;
      const distance = Math.sqrt(rawDistanceSquared);
      const force = repulsionStrength * REPULSION_UNIT_SCALE
        * leftRepulsor.charge * rightRepulsor.charge
        / (rawDistanceSquared + REPULSION_SOFTENING_SQUARED);
      const forceX = dx / distance * force;
      const forceY = dy / distance * force;
      const aComponent = componentByParticle.get(a.id);
      const bComponent = componentByParticle.get(b.id);
      if (
        aComponent !== undefined &&
        bComponent !== undefined &&
        aComponent !== bComponent
      ) {
        externalForces[aComponent].x -= forceX * EXTERNAL_STRUCTURE_REPULSION;
        externalForces[aComponent].y -= forceY * EXTERNAL_STRUCTURE_REPULSION;
        externalForces[bComponent].x += forceX * EXTERNAL_STRUCTURE_REPULSION;
        externalForces[bComponent].y += forceY * EXTERNAL_STRUCTURE_REPULSION;
      } else {
        const aForce = forces.get(a.id);
        const bForce = forces.get(b.id);
        if (aForce) {
          aForce.repulsion.x -= forceX;
          aForce.repulsion.y -= forceY;
        }
        if (bForce) {
          bForce.repulsion.x += forceX;
          bForce.repulsion.y += forceY;
        }
      }
    }
  }

  components.forEach((component, componentIndex) => {
    const componentBodies = [...component]
      .map((id) => bodies.get(id))
      .filter((body): body is Body => body !== undefined);
    if (!componentBodies.length) return;
    const force = externalForces[componentIndex];
    const forceX = force.x / componentBodies.length;
    const forceY = force.y / componentBodies.length;
    componentBodies.forEach((body) => {
      const bodyForce = forces.get(body.id);
      if (!bodyForce) return;
      bodyForce.repulsion.x += forceX;
      bodyForce.repulsion.y += forceY;
    });
  });

  forces.forEach((force) => {
    force.net.x = force.attraction.x + force.repulsion.x;
    force.net.y = force.attraction.y + force.repulsion.y;
  });
  return forces;
}

export function integrate(
  bodies: Map<string, Body>,
  deltaSeconds: number,
  pointerTarget: PointerTarget | null,
  attractions: PhysicsAttraction[],
  repulsors: PhysicsRepulsor[],
  components: Set<string>[],
  parameters: PhysicsParameters,
): PositionSnapshot {
  const previous = new Map<string, Vector>();
  const forces = calculateForces(
    bodies,
    attractions,
    repulsors,
    components,
    parameters.repulsionStrength,
  );

  bodies.forEach((body) => {
    previous.set(body.id, { x: body.x, y: body.y });
    const force = forces.get(body.id);
    if (force) {
      body.vx += force.net.x * deltaSeconds;
      body.vy += force.net.y * deltaSeconds;
    }

    if (pointerTarget?.particleId === body.id) {
      body.x = pointerTarget.x;
      body.y = pointerTarget.y;
      return;
    }

    const freeDecay = Math.exp(-parameters.damping * deltaSeconds);
    body.vx *= freeDecay;
    body.vy *= freeDecay;

    const speed = Math.hypot(body.vx, body.vy);
    if (speed < SETTLE_SPEED) {
      body.vx = 0;
      body.vy = 0;
      return;
    }
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
    if (correction?.x) body.vx = 0;
    if (correction?.y) body.vy = 0;
  });
}
