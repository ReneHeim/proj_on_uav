# Preprocessing scripts

The public scripts are generic helpers for MicaSense RedEdge-P stacks and
orthorectification. Production package behavior belongs in
`src/oncerco_uav/`; scripts should accept paths and options through their CLI.

The RedEdge-P output convention is five bands in this order:

```text
Blue, Green, Red, Red edge, NIR
```

Reflectance stack values use `uint16` storage with
`reflectance = pixel_value / 32767`.
