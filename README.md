# LPBF 3 wt.% Ni-Modified SDSS 2507 Hardness and Tensile Results

This repository contains hardness and tensile-test analysis workflows, scripts, notebooks, and figures for laser powder bed fusion (LPBF)-fabricated 3 wt.% Ni-modified super duplex stainless steel 2507.

## Scope

The repository supports mechanical-property analysis of LPBF-fabricated 3 wt.% Ni-modified SDSS 2507, including Rockwell A hardness, Vickers microhardness, and tensile stress–strain response across the investigated processing and post-processing conditions.

The processing conditions included in the uploaded figures are:

- AS
- SR400
- SR450
- SR500
- SR550
- SA1100

## Repository structure

```text
LPBF_AISI-2507-Hardness-and-Tensile-results/
├── README.md
├── LICENSE
├── figures/
│   ├── AS_experimental_mean_styled.png
│   ├── SR400_experimental_mean_styled.png
│   ├── SR450_experimental_mean_styled.png
│   ├── SR500_experimental_mean_styled.png
│   ├── SR550_experimental_mean_styled.png
│   ├── SA1100_experimental_mean_styled.png
│   └── hardness_gradient_tolmuted.png
├── hardness/
│   ├── Hardness.py
│   └── Hardness.ipynb
└── tensile/
    ├── tensile.py
    └── tensile.ipynb
