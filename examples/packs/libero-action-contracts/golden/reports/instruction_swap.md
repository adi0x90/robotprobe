# Golden Root-Cause Report: Instruction Swap

The instruction-swap incident localizes to the task/instruction boundary. It demonstrates that
language/input integrity is a separate contract from observation freshness and action postprocess.

Expected diagnosis:

- Layer: `instruction`
- Contract: `InstructionIntentContract`
- Fault: `instruction_swap`
- Instruction changed: `true`

Non-claim: this does not prove physical exploitability. It represents a recorded instruction
integrity incident suitable for regression checks.
