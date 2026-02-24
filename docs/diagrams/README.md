# Experiment Diagrams

This folder contains Mermaid source files for experiment workflows and analysis outputs.

## Files

- `experiment_protocol.mmd`: End-to-end experiment protocol from calibration through lead-time outputs.
- `lead_time_methods.mmd`: Threshold-based lead-time detection method used in active analysis.
- `experiment_outputs.mmd`: Experiment artifacts, figures, and analysis summary flow.

## Export to SVG or PNG

If you have Mermaid CLI installed (`mmdc`):

```bash
mmdc -i docs/diagrams/experiment_protocol.mmd -o docs/diagrams/experiment_protocol.svg
mmdc -i docs/diagrams/lead_time_methods.mmd -o docs/diagrams/lead_time_methods.svg
mmdc -i docs/diagrams/experiment_outputs.mmd -o docs/diagrams/experiment_outputs.svg
```
