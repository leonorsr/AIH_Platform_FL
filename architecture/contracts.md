# Proposed contracts

These contracts will be implemented in C# and shared by local simulation and distributed transport. Parameters are tensors identified by name, shape, and data type; a schema hash ensures compatibility.

| Contract | Contents |
| --- | --- |
| `ExperimentConfig` | Experiment ID, seeds, model, classes, clients, rounds, training configuration, and paths |
| `ModelState` | Version, schema hash, and model parameters |
| `TrainRequest` | Experiment, round, client, global version, initial model, epochs, batch size, learning rate, and local seed |
| `ClientUpdate` | Experiment, round, client, received global version, final parameters, distinct-example count, completed steps, and local loss |
| `EvaluationRequest` | Experiment, model version, client, and development/test split |
| `EvaluationResult` | Client, version, example count, loss sum, and a 4×4 confusion matrix in N/S/V/Q order |
| `RoundResult` | Round, selected/accepted clients, example counts and aggregation weights, development metrics, and checkpoint |

## Planned interfaces

- `IFederatedClient.TrainAsync`: receives instructions, trains locally, and returns an update.
- `IFederatedEvaluator.EvaluateAsync`: evaluates without modifying parameters and returns counts and metrics.
- `IModelStateCodec`: extracts and applies parameters while validating the schema.
- `IAggregator.Aggregate`: receives compatible updates and returns a new global state without modifying its inputs.
- `IClientSelector.Select`: selects only client IDs allowed in the training split.
- `ICheckpointStore`: publishes and reads complete checkpoints and the best-development-checkpoint record.

## Validation before aggregation

The experiment, round, and version must match the request. Each client may respond only once and must belong to the training split. The server checks `n_k > 0` and, for the baseline, equality with the recorded count for that client. Parameter names, types, and dimensions must match the global model; every value must be finite.

Aggregation fails when there are no responses, the total weight is zero, or a response is invalid. The global state advances only after all expected responses are valid and the checkpoint has been published. Stale updates cannot enter a new round.

## Acceptance checks

1. Two clients with scalar parameters 2 and 8 and example counts 1 and 3 produce 6.5.
2. A single client reproduces its state exactly; response order does not change the result beyond the specified numerical tolerance.
3. Duplicate responses, incompatible shapes, NaN/Infinity, zero counts, and incorrect versions are rejected.
4. Development/test clients never receive training requests or participate in aggregation.
5. All round participants start from the same checkpoint with independent local states.
6. Global evaluation sums confusion matrices and losses while preserving per-class support.
7. A mid-round failure preserves the last completed checkpoint; recovery reproduces the next round's seeds and selection.
