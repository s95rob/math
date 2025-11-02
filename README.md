# Math

A (mostly) efficient, (mostly) linear algebra library in C++11. Header-only. 

***NOTE:*** Matrices are stored in row-major order for memory efficiency. Before sending them to any GAPI that uses column-major matrices (D3D, OpenGL, etc.), transpose them using `Math::Transpose`.