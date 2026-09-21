# markers/

Printable sheets for the real-footage check. Committed rather than generated, so
everyone prints the same files.

| | |
| --- | --- |
| Family | `DICT_4X4_50`, the dictionary `scene.aruco_dictionary` returns |
| Marker side | 150 mm, black border included |
| Quiet zone | 18.75 mm of white on every side |
| Page | US Letter, portrait |

One marker and its id per sheet. Nothing else is printed, because the page is in frame
in every photograph taken with it.

The quiet zone is the `side / 8` ratio `scene.render` pads by. Cutting flush to the
black border measures a failure mode the sweep never generated.

## Printing

Print at 100%, no fit-to-page and no toner saving: the markers are pure black and white,
and a washed black changes the contrast adaptive thresholding sees. Matte paper, not
glossy, since specular highlights are another condition the sweep does not model.

Check the scale by measuring the black square. It has to read 150 mm.

## Framing

The sweep renders a marker at 96 px in a 640x480 frame, 15% of the frame width. To match
it, the scene visible at the marker's plane has to be 6.67 times the marker side, so
1.00 m wide for these sheets.

Framing by scene width instead of by camera distance is on purpose: the distance that
produces it depends on the lens.
