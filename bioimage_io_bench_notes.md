# bioimageio bench notes

## Design

### Features

- model suggestion
  - based on tags
  - based on user images
- model ranking

## Implementation

### Updates

#### Dataset selection

- [ ] data inputs support multiple data files (samples) in one dataset group
- [x] support glob pattern/directory for multi file dataset selection
- [x] New data -> add dataset (button to pop-up/not fully extended initially)
- [x] editable dataset name
- [ ] support multi tensor inputs
- [x] for multiple samples from a single dataset add a button to allow roi selection settings from first image to be auto applied to all samples. "sync ROI selection"

#### Model selection

- [ ] add model button adds model id/name to the current model selection list/table that includes model id and model name and the blockwise checkbox
- [ ] support comma separated adding of multiple models at once

#### Prediction comparison

- [ ] slice selection slider for viewer per dataset view
  - [ ] sync input view sliders with output view sliders for common dimensions
- [ ] add input tensors to the comparison matrix as the first column(s)
- [ ] support multi tensor outputs
