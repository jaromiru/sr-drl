(define (domain boxworld)
(:requirements :strips)
(:constants GROUND)
(:predicates (box-on ?x ?y) ; ?x is on ?y
             (free ?x) ; nothing is on top of ?x
)

(:action move
 :parameters (?base ?what ?where)
 :precondition (and (not (= ?what ?where))
 					(free ?what)
                    (or (free ?where) (= ?where GROUND))
                    (box-on ?what ?base))
 :effect (and (not (box-on ?what ?base)) 	; remove ?what from ?base
              (box-on ?what ?where) 		; put ?what on ?where
              (free ?base) 					; we moved ?what so ?base is free
              (not (free ?where)) 			; ?where is not free anymore
         )
)
)

