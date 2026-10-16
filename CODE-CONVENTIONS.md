# Code conventions

Clean as you go. Whenever you touch a file that has violations, fix those violations.

## Abstraction
* AB-e5: A helper with one call site is inlined.
* AB-o4: Branching with switch/match should always be exhaustive.
* AB-k2: No module-level constants. They should be a parameter, or be constants on the class they belong to.

## Architecture
* AT-7b: Infrastructure errors should _not_ be part of port signatures. E.g. `HTTPError`. They should be domain errors.
* AT-f8: Ports should be named after domain needs, not technology. E.g. `MetaAdProvider`, not `ZernioMetaAdProvider`.
* AT-bf: Make side-effects dependency-injectable, like time (use a Clock), randomness (use a random adapter), and persistence (filesystem etc.). This gives you control in tests.
* AT-q4: A function either computes or has effects, never both (CQRS).

## Error handling
* EH-8c: A failure returns a typed error. It never returns `None`, an empty collection, or `False`.
* EH-t1: An error stays in the return type until it is handled.
* EH-v6: Logging a warning is not error handling. If the caller must act on it, return an error.
* EH-g3: An error's name and message should contain the offending values (which IDs were missing), not only that something went wrong.
* EH-n5: Each failure mode returns a distinct error; two causes do not share one error type.
* EH-j8: Validation reports every violation it can see rather than stopping at the first.
* EH-s2: Invalid states should not be representable. If you think a function should raise, consider whether the types could be constrained to make the error impossible.

## Naming
* NA-8f: Function calls without module prefixes are hard to read, because you don't know where the callee lives. E.g. not `send_to_user(item)`, but `package_provider.send(item)`. Instead, add a prefix, either by:
	* Importing the modules, not the functions
	* Disallowing free functions, and using static methods or regular methods
* NA-9h: Use descriptive method names. No `to(param)`, instead `to_<type>(param)`
* NA-f7: Two distinct concepts never share a name.
* NA-y2: A wrapper's name says what it is, not that it wraps.

## Resilience
* RE-37: Every network operation needs retries and a timeout
* RE-a7: Whenever something retries, log the retry and/or update a metric

## Testing
* TE-3a: The arrange step passes an input only when the outcome depends on it.
	* Relations between inputs (same, different, contains, ordered) are expressed in code, not literals.
	* A literal is allowed only when the literal itself is under test.
* TE-8f: Each value object has a constructor for creating fake data.
* TE-hj: Constructors for fake data separate `random_fake` from `static_fake`. Only one is required.
* TE-b7: Cases differing only in their inputs are one parameterised test.
* TE-r2: Test data is created through the same entry point production uses, so it passes the same validation and logic.
* TE-c8: Fakes/simulators should be stateful, and assertions should be against the simulated state.
* TE-m8: Mocking and patching is not allowed. Tests should never assert on implementation.
* TE-9c: Do _not_ maintain a `tests` folder. Instead, place tests next to the code they test, as `test_<module>`, or in the same file when possible.

## Types
* TY-w3: A value with a unit, format or range is its own type (`NonNegativeInt`, `Email`), not a primitive plus validation at each use.
* TY-k8: Avoid booleans and nullable fields. Instead, prefer an enum or a union.
	* Booleans can be hard to read, and even harder to extend.
	* If `None` would carry a meaning, that meaning is an enum variant or an error type.
* TY-d2: A field has no default, except a discriminator tag.
* TY-h5: Datetimes are timezone-aware.
* TY-m1: Parameters accept the widest collection type that works.
* TY-z9: No new lint or type suppressions without the user's approval.
* TY-jd: Prefer immutable data structures whenever possible.
* TY-c1: _Never_ allow primitives as function parameters, fields, etc. Instead, use a Pydantic RootModel.
* TY-a8: On each domain model/rootmodel, add a `.fake` static method for testing. It should hold default values for every value. When the item is an aggregate, call the `.fake` of its member objects to construct the default values.
* TY-7e: Return errors as values using https://github.com/overflowy/safe-result
* TY-fg: Use nominal typing and explicit subtyping. E.g. if a class is implementing a protocol, also make it inherit that protocol.
* TY-8f: Prefer Pydantic models > dataclasses > regular classes.

## Modules
* MO-9r: When a module grows large enough that it needs multiple files, place it in a folder with its own `__init__` for exports.
* MO-t8: _Never_ maintain a `__all__` list. Instead, use direct imports.
* MO-kf: Avoid constants. Whenever you need a constant, consider whether it should be an argument for the caller instead.
