## C.1 — token_service_fraction por tenant

Formula: tokens_completados_i / (peticiones_i * NPRED), con NPRED=128 fijo en todas
las corridas multi-tenant. Fuente: llama_fairness_por_tenant.csv, columnas
`completed_output_tokens` y `offered_output_tokens`.

| sistema | N | lambda | rep | est | tsf peor | tsf mediana | tsf mejor | Jain tsf | tsf peor/mejor |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| gusto | 1 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto | 2 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto | 2 | 2.0 | 1 | U | 0.9412 | 0.9706 | 1.0000 | 0.9991 | 1.06 |
| gusto | 2 | 2.0 | 2 | U | 0.9412 | 0.9706 | 1.0000 | 0.9991 | 1.06 |
| gusto | 2 | 2.0 | 3 | U | 0.9412 | 0.9706 | 1.0000 | 0.9991 | 1.06 |
| gusto | 4 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto | 4 | 4.0 | 1 | U | 0.3478 | 0.4119 | 0.4211 | 0.9946 | 1.21 |
| gusto | 4 | 4.0 | 2 | U | 0.3617 | 0.3884 | 0.4474 | 0.993 | 1.24 |
| gusto | 4 | 4.0 | 3 | U | 0.3617 | 0.3812 | 0.4737 | 0.988 | 1.31 |
| gusto | 8 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto | 8 | 8.0 | 1 | U | 0.1702 | 0.2239 | 0.3226 | 0.9568 | 1.90 |
| gusto | 8 | 8.0 | 2 | U | 0.1702 | 0.2212 | 0.2903 | 0.9679 | 1.71 |
| gusto | 8 | 8.0 | 3 | U | 0.1489 | 0.2192 | 0.3448 | 0.9303 | 2.32 |
| gusto_gpudirect | 1 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 2 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 2 | 1.0 | 2 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 2 | 1.0 | 3 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 2 | 1.0 | 4 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 2 | 1.0 | 5 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 4 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 4 | 1.0 | 2 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 4 | 1.0 | 3 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 6 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 8 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 8 | 1.0 | 2 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| gusto_gpudirect | 10 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| nativo | 1 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| nativo | 2 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| nativo | 2 | 2.0 | 1 | U | 0.7714 | 0.7975 | 0.8235 | 0.9989 | 1.07 |
| nativo | 2 | 2.0 | 2 | U | 0.7714 | 0.7975 | 0.8235 | 0.9989 | 1.07 |
| nativo | 2 | 2.0 | 3 | U | 0.7714 | 0.7975 | 0.8235 | 0.9989 | 1.07 |
| nativo | 4 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| nativo | 4 | 4.0 | 1 | U | 0.2766 | 0.3025 | 0.3333 | 0.9954 | 1.21 |
| nativo | 4 | 4.0 | 2 | U | 0.2766 | 0.3025 | 0.3333 | 0.9954 | 1.21 |
| nativo | 4 | 4.0 | 3 | U | 0.2766 | 0.3025 | 0.3333 | 0.9954 | 1.21 |
| nativo | 8 | 1.0 | 1 | S | 1.0000 | 1.0000 | 1.0000 | 1.0 | 1.00 |
| nativo | 8 | 8.0 | 1 | U | 0.1277 | 0.1614 | 0.2258 | 0.9602 | 1.77 |
| nativo | 8 | 8.0 | 2 | U | 0.1277 | 0.1614 | 0.2258 | 0.9602 | 1.77 |
| nativo | 8 | 8.0 | 3 | U | 0.1277 | 0.1614 | 0.2258 | 0.9602 | 1.77 |

## C.2 — Jain de SLO: donde tiene sentido y donde no

| sistema | N | lambda | rep | tenants con SLO>0 | atencion media | Jain SLO 5 s |
|---|---:|---:|---:|---:|---:|---|
| gusto | 1 | 1.0 | 1 | 1/1 | 1.000 | 1.0000 |
| gusto | 2 | 1.0 | 1 | 2/2 | 1.000 | 1.0000 |
| gusto | 2 | 2.0 | 1 | 2/2 | 0.164 | 0.9890 |
| gusto | 2 | 2.0 | 2 | 2/2 | 0.164 | 0.9890 |
| gusto | 2 | 2.0 | 3 | 2/2 | 0.164 | 0.9890 |
| gusto | 4 | 1.0 | 1 | 4/4 | 1.000 | 1.0000 |
| gusto | 4 | 4.0 | 1 | 1/4 | 0.005 | **SIN POTENCIA** (inanicion: 1/4 tenants, media 0.005) |
| gusto | 4 | 4.0 | 2 | 2/4 | 0.012 | **SIN POTENCIA** (inanicion: 2/4 tenants, media 0.012) |
| gusto | 4 | 4.0 | 3 | 1/4 | 0.005 | **SIN POTENCIA** (inanicion: 1/4 tenants, media 0.005) |
| gusto | 8 | 1.0 | 1 | 8/8 | 1.000 | 1.0000 |
| gusto | 8 | 8.0 | 1 | 5/8 | 0.016 | **SIN POTENCIA** (inanicion: 5/8 tenants, media 0.016) |
| gusto | 8 | 8.0 | 2 | 5/8 | 0.017 | **SIN POTENCIA** (inanicion: 5/8 tenants, media 0.017) |
| gusto | 8 | 8.0 | 3 | 4/8 | 0.013 | **SIN POTENCIA** (inanicion: 4/8 tenants, media 0.013) |
| gusto_gpudirect | 1 | 1.0 | 1 | 1/1 | 1.000 | 1.0000 |
| gusto_gpudirect | 2 | 1.0 | 1 | 2/2 | 1.000 | 1.0000 |
| gusto_gpudirect | 2 | 1.0 | 2 | 2/2 | 1.000 | 1.0000 |
| gusto_gpudirect | 2 | 1.0 | 3 | 2/2 | 1.000 | 1.0000 |
| gusto_gpudirect | 2 | 1.0 | 4 | 2/2 | 1.000 | 1.0000 |
| gusto_gpudirect | 2 | 1.0 | 5 | 2/2 | 1.000 | 1.0000 |
| gusto_gpudirect | 4 | 1.0 | 1 | 4/4 | 1.000 | 1.0000 |
| gusto_gpudirect | 4 | 1.0 | 2 | 4/4 | 1.000 | 1.0000 |
| gusto_gpudirect | 4 | 1.0 | 3 | 4/4 | 1.000 | 1.0000 |
| gusto_gpudirect | 6 | 1.0 | 1 | 6/6 | 1.000 | 1.0000 |
| gusto_gpudirect | 8 | 1.0 | 1 | 8/8 | 1.000 | 1.0000 |
| gusto_gpudirect | 8 | 1.0 | 2 | 8/8 | 1.000 | 1.0000 |
| gusto_gpudirect | 10 | 1.0 | 1 | 10/10 | 1.000 | 1.0000 |
| nativo | 1 | 1.0 | 1 | 1/1 | 1.000 | 1.0000 |
| nativo | 2 | 1.0 | 1 | 2/2 | 1.000 | 1.0000 |
| nativo | 2 | 2.0 | 1 | 1/2 | 0.059 | 0.5000 |
| nativo | 2 | 2.0 | 2 | 1/2 | 0.059 | 0.5000 |
| nativo | 2 | 2.0 | 3 | 1/2 | 0.059 | 0.5000 |
| nativo | 4 | 1.0 | 1 | 4/4 | 1.000 | 1.0000 |
| nativo | 4 | 4.0 | 1 | 2/4 | 0.017 | **SIN POTENCIA** (inanicion: 2/4 tenants, media 0.017) |
| nativo | 4 | 4.0 | 2 | 2/4 | 0.017 | **SIN POTENCIA** (inanicion: 2/4 tenants, media 0.017) |
| nativo | 4 | 4.0 | 3 | 2/4 | 0.017 | **SIN POTENCIA** (inanicion: 2/4 tenants, media 0.017) |
| nativo | 8 | 1.0 | 1 | 8/8 | 0.984 | 0.9982 |
| nativo | 8 | 8.0 | 1 | 3/8 | 0.012 | **SIN POTENCIA** (inanicion: 3/8 tenants, media 0.012) |
| nativo | 8 | 8.0 | 2 | 3/8 | 0.012 | **SIN POTENCIA** (inanicion: 3/8 tenants, media 0.012) |
| nativo | 8 | 8.0 | 3 | 3/8 | 0.012 | **SIN POTENCIA** (inanicion: 3/8 tenants, media 0.012) |

## C.3 — comparacion emparejada Gusto frente a nativo

Pares emparejados por (N, lambda, repeticion): **13**.

| N | lambda | rep | est | Jain compl. Gusto | nativo | dif | compl. media Gusto | nativo | dif |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 1 | S | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| 2 | 1.0 | 1 | S | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| 2 | 2.0 | 1 | U | 0.9991 | 0.9989 | +0.0002 | 0.9706 | 0.7975 | +0.1731 |
| 2 | 2.0 | 2 | U | 0.9991 | 0.9989 | +0.0002 | 0.9706 | 0.7975 | +0.1731 |
| 2 | 2.0 | 3 | U | 0.9991 | 0.9989 | +0.0002 | 0.9706 | 0.7975 | +0.1731 |
| 4 | 1.0 | 1 | S | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| 4 | 4.0 | 1 | U | 0.9946 | 0.9954 | -0.0008 | 0.3982 | 0.3037 | +0.0944 |
| 4 | 4.0 | 2 | U | 0.9930 | 0.9954 | -0.0024 | 0.3965 | 0.3037 | +0.0928 |
| 4 | 4.0 | 3 | U | 0.9880 | 0.9954 | -0.0074 | 0.3994 | 0.3037 | +0.0957 |
| 8 | 1.0 | 1 | S | 1.0000 | 1.0000 | +0.0000 | 1.0000 | 1.0000 | +0.0000 |
| 8 | 8.0 | 1 | U | 0.9568 | 0.9602 | -0.0034 | 0.2366 | 0.1675 | +0.0691 |
| 8 | 8.0 | 2 | U | 0.9679 | 0.9602 | +0.0077 | 0.2255 | 0.1675 | +0.0579 |
| 8 | 8.0 | 3 | U | 0.9303 | 0.9602 | -0.0299 | 0.2321 | 0.1675 | +0.0645 |

## C.4 — intervalos de confianza y equivalencia

- **Jain de completion fraction** (Gusto menos nativo), n=13 pares: media **-0.0027**, IC95 bootstrap **[-0.0079, +0.0009]**.
  - TOST con margen +-0.05: **EQUIVALENTES** (el IC cabe dentro de [-0.05, +0.05]).
  - Potencia: con n=13 pares el test es debil; un IC que quepa en el margen es evidencia de equivalencia, uno que no quepa NO es evidencia de diferencia.
- **completion fraction media** (Gusto menos nativo), n=13 pares: media **+0.0764**, IC95 bootstrap **[+0.0426, +0.1122]**.
