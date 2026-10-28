# Breakeven Demonstration of qLDPC Codes
This is a companion repository for the paper "Breakeven demonstration of quantum low-density parity-check codes" (arXiv:2606.06455).

# License
The content in this repository is copyright IonQ Inc., all rights reserved. It is licensed under the [CC BY-NC-SA 4.0 license](LICENSE.md).

# Usage
Decoding of experimental data is shown in "Analyze_Data.ipynb".
Pre-computed, aggregated decoding outcomes are saved in "processed_data/".

Main plots and results tables in the manuscript are generated in "Plot_Data.ipynb".
These utilize outputs of "Analyze_Data.ipynb".

# External Dependency
Decoding of data relies on the Beam Search decoder, described in [Ye et al. PRX Quantum 7, 033002 (2026)](https://doi.org/10.1103/6k5x-ztqt), and publicly available [here](https://github.com/ionq-publications/BeamSearchDecoder).

Note, by default scripts here call that publicly available decoder.
But for consistency and completeness we also include an internal (slightly faster) version of Beam Search as an object file, that implements the exact version of the decoder used during preparation of our manuscript.
Users can choose to call it by instantiating ``decoder_constructor(decname="beam_internal")``.


# Contact
Please direct questions / comments to tham@ionq.co