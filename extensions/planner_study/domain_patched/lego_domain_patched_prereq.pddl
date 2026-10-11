;; PATCHED COPY (positive preconditions only) of src/workbenchmark_pddl/pddl/lego_domain.pddl
;; (planner study, phase 4). The original is not modified. Changes are marked PATCH.
;;
;; The rule: a brick that hangs over another brick must be placed AFTER it (it could
;; not be lowered into place otherwise). Stated as a prerequisite: place and stack
;; require that every brick the placed brick hangs over is already in the assembly
;; area. For complete plans (every brick ends up placed) this allows exactly the
;; orderings that "no brick that hangs over ?b may be placed before ?b" allows, but it
;; has no dead ends and needs no negation, quantifier or delete effect, so every
;; engine (A* + LM-cut, pyperplan) can use it and heuristics see the ordering.
;;   (hangs-over ?c ?d1 .. ?d6)   static, from the GOAL layout: ?c hangs over ?d1..?d6,
;;                                padded with the dummy brick b_dummy, which the
;;                                problem declares as already in the assembly area
;; The 6 slots are trailing parameters of place and stack, bound by the static fact
;; (one grounding per brick); they are dropped before a plan is executed
;; (patched_pddl.strip_slots).
;; --- original header ---
;; LEGO Duplo assembly domain for WorkBenchMark (Project P1)
;; Stud-grid model with pose = (anchor stud, layer, rotation).
;; Aimed for Tasks up to Tier 3 (vertical stacks, half-overlap, multi-column)

(define (domain lego-assembly)
  (:requirements :strips :typing)

  (:types
    ;; Bricks are the physical objects in the world
    brick

    ;; Brick types from the catalog, e.g. 2x2 / 4x2
    brick-type 

    ;; Robot end-effector (gripper)
    endeffector

    ;; Studs are the discrete XY cells on the Duplo pitch (16 mm)
    stud

    ;; Layers are the discrete Z levels (brick height ~19.1 mm)
    layer

    ;; Rotations on the grid (0° / 90° in the dataset)
    rot

    ;; Areas are the pick vs assembly workspace
    area
  )

  (:predicates

    ;; Bricks are at a stud, layer, and oriented at a rotation
    (at-stud ?b - brick ?s - stud)
    (at-layer ?b - brick ?l - layer)
    (oriented ?b - brick ?r - rot)

    ;; Bricks are in a area
    (in-area ?b - brick ?a - area)

    ;; Bricks are clear (if no other bricks stacked on top)
    (clear ?b - brick)

    ;; Bricks support other bricks (allow multiple bottoms (bridges))
    (supports ?bottom - brick ?top - brick)

    ;; Gripper is holding a brick
    (holding ?g - endeffector ?b - brick)

    ;; Gripper is empty
    (gripper-empty ?g - endeffector)

    ;; Only the brick's anchor stud is marked occupied/free
    (occupied ?s - stud ?l - layer)
    (free ?s - stud ?l - layer)

    ;; Bricks are of a type
    (is-type ?b - brick ?t - brick-type)

    ;; PATCH (static): ?c hangs over ?d1 .. ?d6; padded with b_dummy
    (hangs-over ?c - brick ?d1 - brick ?d2 - brick ?d3 - brick ?d4 - brick ?d5 - brick ?d6 - brick)

    ;; Areas are pick vs assembly
    (area-pick ?a - area)
    (area-assembly ?a - area)

    ;; Base layer is the lowest layer
    (base-layer ?l - layer)                

    ;; Layers are stacked above each other
    (above ?upper - layer ?lower - layer) 

    ;; Type and rotation with anchor at ?anchor covers ?cell.
    (footprint ?t - brick-type ?r - rot ?anchor - stud ?cell - stud)

    ;; True when top/bottom footprints overlap enough to stud-connect at those anchors
    (can-attach
      ?t-top - brick-type ?r-top - rot ?top-anchor - stud
      ?t-bot - brick-type ?r-bot - rot ?bot-anchor - stud)
  )

  ;; Grasp a clear brick from the pick area (poses come from YAML / perception).
  (:action pick
    :parameters (
      ?g - endeffector
      ?b - brick
      ?s - stud
      ?l - layer
      ?r - rot
      ?a - area
      ?t - brick-type
    )
    :precondition (and
      (gripper-empty ?g)
      (clear ?b)
      (at-stud ?b ?s)
      (at-layer ?b ?l)
      (oriented ?b ?r)
      (in-area ?b ?a)
      (area-pick ?a)
      (is-type ?b ?t)
      (occupied ?s ?l)
    )
    :effect (and
      (holding ?g ?b)
      (free ?s ?l)
      (not (gripper-empty ?g))
      (not (at-stud ?b ?s))
      (not (at-layer ?b ?l))
      (not (oriented ?b ?r))
      (not (in-area ?b ?a))
      (not (clear ?b))
      (not (occupied ?s ?l))
    )
  )

  ;; Put a held brick onto a free assembly base-layer anchor.
  (:action place
    :parameters (
      ?g - endeffector
      ?b - brick
      ?s - stud
      ?l - layer
      ?r - rot
      ?a - area
      ?t - brick-type
      ?d1 - brick
      ?d2 - brick
      ?d3 - brick
      ?d4 - brick
      ?d5 - brick
      ?d6 - brick ;; PATCH: slots, bound by (hangs-over ?b ...)
    )
    :precondition (and
      (holding ?g ?b)
      (area-assembly ?a)
      (base-layer ?l)
      (is-type ?b ?t)
      (free ?s ?l)
      (hangs-over ?b ?d1 ?d2 ?d3 ?d4 ?d5 ?d6) ;; PATCH
      (in-area ?d1 ?a)
      (in-area ?d2 ?a)
      (in-area ?d3 ?a)
      (in-area ?d4 ?a)
      (in-area ?d5 ?a)
      (in-area ?d6 ?a) ;; PATCH: everything ?b hangs over is already placed
    )
    :effect (and
      (gripper-empty ?g)
      (at-stud ?b ?s)
      (at-layer ?b ?l)
      (oriented ?b ?r)
      (in-area ?b ?a)
      (clear ?b)
      (occupied ?s ?l)
      (not (holding ?g ?b))
      (not (free ?s ?l))
    )
  )

  ;; Stack when can-attach holds; bottom stays clear so several tops can share it.
  ;; Same top cell is still blocked by free/occupied on (?top-s, ?top-l).
  (:action stack
    :parameters (
      ?g - endeffector
      ?top - brick
      ?bottom - brick
      ?top-s - stud
      ?bot-s - stud
      ?top-l - layer
      ?bot-l - layer
      ?top-r - rot
      ?bot-r - rot
      ?a - area
      ?t-top - brick-type
      ?t-bot - brick-type
      ?d1 - brick
      ?d2 - brick
      ?d3 - brick
      ?d4 - brick
      ?d5 - brick
      ?d6 - brick ;; PATCH: slots, bound by (hangs-over ?top ...)
    )
    :precondition (and
      (holding ?g ?top)
      (at-stud ?bottom ?bot-s)
      (at-layer ?bottom ?bot-l)
      (oriented ?bottom ?bot-r)
      (in-area ?bottom ?a)
      (area-assembly ?a)
      (is-type ?top ?t-top)
      (is-type ?bottom ?t-bot)
      (above ?top-l ?bot-l)
      (can-attach ?t-top ?top-r ?top-s ?t-bot ?bot-r ?bot-s)
      (free ?top-s ?top-l)
      (hangs-over ?top ?d1 ?d2 ?d3 ?d4 ?d5 ?d6) ;; PATCH
      (in-area ?d1 ?a)
      (in-area ?d2 ?a)
      (in-area ?d3 ?a)
      (in-area ?d4 ?a)
      (in-area ?d5 ?a)
      (in-area ?d6 ?a) ;; PATCH: everything ?top hangs over is already placed
    )
    :effect (and
      (gripper-empty ?g)
      (at-stud ?top ?top-s)
      (at-layer ?top ?top-l)
      (oriented ?top ?top-r)
      (in-area ?top ?a)
      (supports ?bottom ?top)
      (clear ?top)
      (occupied ?top-s ?top-l)
      (not (holding ?g ?top))
      (not (free ?top-s ?top-l))
    )
  )
)
