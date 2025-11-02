#pragma once

#include <cmath>

namespace Math {
    
    /* Constants */

    template <typename T>
    constexpr T PI = static_cast<T>(3.1415926535);


    /* Trigonometry */

    template <typename T>
    inline T Radians(T degrees) {
        return degrees * (PI<T> / static_cast<T>(180.0));
    }

    template <typename T>
    inline T Degrees(T radians) {
        return radians * (static_cast<T>(180.0) / PI<T>);
    }


    /* Vector math */

    template <typename T, int N>
    struct VectorStorage {
        T Elements[N];
        };
    
    template <typename T>
    struct VectorStorage<T, 2> {
        union {
            T Elements[2];
            struct {
                T X, Y;
            };
        };
        
        VectorStorage() = default;
        VectorStorage(T xy)
        : VectorStorage(xy, xy) {}
        VectorStorage(T x, T y)
        : X(x), Y(y) {}
    };

    template <typename T>
    struct VectorStorage<T, 3> {
        union {
            T Elements[3];
            struct {
                T X, Y, Z;
            };
        };
        
        VectorStorage() = default;
        VectorStorage(T xyz)
        : VectorStorage(xyz, xyz, xyz) {}
        VectorStorage(T x, T y, T z)
        : X(x), Y(y), Z(z) {}
    };

    template <typename T>
    struct VectorStorage<T, 4> {
        union {
            T Elements[4];
            struct {
                T X, Y, Z, W;
            };
        };
        
        VectorStorage() = default;
        VectorStorage(T xyz)
        : VectorStorage(xyz, xyz, xyz, static_cast<T>(1.0)) {}
        VectorStorage(T xyz, T w)
        : VectorStorage(xyz, xyz, xyz, w) {}
        VectorStorage(T x, T y, T z)
        : VectorStorage(x, y, z, static_cast<T>(1.0)) {}
        VectorStorage(T x, T y, T z, T w)
        : X(x), Y(y), Z(z), W(w) {}
    };

    template <typename T, int N>
    struct Vector : VectorStorage<T, N> {
        using VectorType = Vector<T, N>;
        
        using VectorStorage<T, N>::VectorStorage;
        Vector() = default;
        
        Vector(const Vector& other)
        : VectorStorage(other) {}

        /* Scalar arithmetic operators */
        
        Vector operator+(T scalar) const {
            Vector v;
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] + scalar;
            
            return v;
        }
        
        Vector operator-(T scalar) const {
            Vector v;
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] - scalar;
            
            return v;
        }
        
        Vector operator*(T scalar) const {
            Vector v;
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] * scalar;
            
            return v;
        }
        
        Vector operator/(T scalar) const {
            Vector v;
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] / scalar;
        }
        
        Vector& operator+=(T scalar) {
            for (int i = 0; i < N; i++)
            (*this)[i] += scalar;
            
            return *this;
        }
        
        Vector& operator-=(T scalar) {
            for (int i = 0; i < N; i++)
            (*this)[i] -= scalar;
            
            return *this;
        }
        
        Vector operator*=(T scalar) {
            for (int i = 0; i < N; i++)
            (*this)[i] *= scalar;
            
            return *this;
        }
        
        Vector operator/=(T scalar) {
            for (int i = 0; i < N; i++)
            (*this)[i] /= scalar;
            
            return *this;
        }
        
        /* Vector arithmetic operators */
        
        Vector operator+(Vector v) const {
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] + v[i];
            
            return v;
        }
        
        Vector operator-(Vector v) const {
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] - v[i];
            
            return v;
        }
        
        Vector operator*(Vector v) const {
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] * v[i];
            
            return v;
        }
        
        Vector operator/(Vector v) const {
            for (int i = 0; i < N; i++)
            v[i] = (*this)[i] / v[i];
            
            return v;
        }
        
        Vector& operator+=(const Vector& v) {
            for (int i = 0; i < N; i++)
            (*this)[i] += v[i];
            
            return *this;
        }
        
        Vector& operator-=(const Vector& v) {
            for (int i = 0; i < N; i++)
            (*this)[i] -= v[i];
            
            return *this;
        }
        
        Vector& operator*=(const Vector& v) {
            for (int i = 0; i < N; i++)
            (*this)[i] *= v[i];
            
            return *this;
        }
        
        Vector& operator/=(const Vector& v) {
            for (int i = 0; i < N; i++)
            (*this)[i] /= v[i];
            
            return *this;
        }
        
        /* Access operators */
        
        T& operator[](int i) { return this->Elements[i]; }
        const T& operator[](int i) const { return this->Elements[i]; }
    };

    using Vector2 = Vector<float, 2>;
    using Vector3 = Vector<float, 3>;
    using Vector4 = Vector<float, 4>;
    
    using Vector2p = Vector<double, 2>;
    using Vector3p = Vector<double, 2>;
    using Vector4p = Vector<double, 2>;
    

    /* Matrix math */
    
    template <typename T, int NRows, int NColumns>
    struct Matrix {
        using VectorType = Vector<T, NColumns>;
        using MatrixType = Matrix<T, NRows, NColumns>;
        
        Matrix() = default;
        Matrix(T scalar) {
            for (int i = 0; i < NRows; i++)
                for (int j = 0; j < NColumns; j++)
                    Elements[i][j] = (i == j)
                        ? scalar
                        : static_cast<T>(0.0);
            }
                    
        /* Scalar arithmetic operators */
        
        Matrix operator+(T scalar) const {
            Matrix r;
            
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            r.Elements[i] = Elements[i] + scalar;
            
            return r;
        }
        
        Matrix operator-(T scalar) const {
            Matrix r;
            
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            r.Elements[i] = Elements[i] - scalar;
            
            return r;
        }
        
        Matrix operator*(T scalar) const {
            Matrix r;
            
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            r.Elements[i] = Elements[i] * scalar;
            
            return r;
        }
        
        Matrix operator/(T scalar) const {
            Matrix r;
            
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            r.Elements[i] = Elements[i] / scalar;
            
            return r;
        }
        
        Matrix operator+=(T scalar) {
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            Elements[i] += scalar;
            
            return *this;
        }
        
        Matrix operator-=(T scalar) {
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            Elements[i] -= scalar;
            
            return *this;
        }
        
        Matrix operator*=(T scalar) {
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            Elements[i] *= scalar;
            
            return *this;
        }
        
        Matrix operator/=(T scalar) {
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            Elements[i] /= scalar;
            
            return *this;
        }
        
        /* Matrix arithmetic operators */
        
        Matrix operator+(const Matrix& m) const {
            Matrix r;
            
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            r.Elements[i] = Elements[i] + m.Elements[i];
            
            return r;
        }
        
        Matrix operator-(const Matrix& m) const {
            Matrix r;
            
            for (int i = 0; i < TOTAL_ELEMENTS; i++)
            r.Elements[i] = Elements[i] - m.Elements[i];
            
            return r;
        }
        
        template <int NRows2, int NColumns2>
        Matrix<T, NRows, NColumns2> operator*(const Matrix<T, NRows2, NColumns2>& m) const {
            static_assert(NColumns == NRows2, "NColumns of this matrix must equal NRows of matrix m");
            
            Matrix<T, NRows, NColumns2> r;
            
            for (int i = 0; i < NRows; i++) {
                for (int j = 0; j < NColumns2; j++) {
                    T dot = static_cast<T>(0.0);
                    
                    for (int k = 0; k < NColumns; k++)
                    dot += (*this)[i][k] * m[k][j];
                    
                    r[i][j] = dot;
                }
            }
            
            return r;
        }
        
        Matrix& operator+=(const Matrix& m) {
            *this = (*this) + m;
            return *this;
        }
        
        Matrix& operator-=(const Matrix& m) {
            *this = (*this) - m;
            return *this;
        }
        
        Matrix& operator*=(const Matrix& m) {
            *this = (*this) * m;
            return *this;
        }
        
        /* Access operators */
        
        VectorType& operator[](int i) { return Elements[i]; }
        const VectorType& operator[](int i) const { return Elements[i]; }
        
        protected:
        VectorType Elements[NRows];
        
        constexpr static int TOTAL_ELEMENTS = NRows * NColumns;
    };
    
    struct Matrix4x4 : Matrix<float, 4, 4> {
        Matrix4x4() = default;
        Matrix4x4(float scalar)
        : MatrixType(scalar) {}
        
        Matrix4x4(Vector4 v0, Vector4 v1, Vector4 v2, Vector4 v3) {
            Elements[0] = v0;
            Elements[1] = v1;
            Elements[2] = v2;
            Elements[3] = v3;
        }
        
        Matrix4x4(const MatrixType& m)
        : MatrixType(m) {}
        
    };

    using Matrix4 = Matrix4x4;
    
    
    /* Vector functions */
    
    // Normalize vector `v`
    template <typename T, int N>
    inline Vector<T, N> Normalize(Vector<T, N> v) {
        const T len = Length(v);
        
        for (int i = 0; i < N; i++)
        v[i] = v[i] / len;
        
        return v;
    }

    // Calculate the length (squared) of vector `v`
    template <typename T, int N>
    inline T LengthSq(const Vector<T, N>& v) {
        T r = static_cast<T>(0.0);

        for (int i = 0; i < N; i++) 
            r += v[i] * v[i];

        return r;
    }

    // Calculate the length of vector `v`
    template <typename T, int N>
    inline T Length(const Vector<T, N>& v) {
        return std::sqrt(LengthSq(v));
    }

    // Calculate the dot product of vectors `v1` and `v2`
    template <typename T, int N>
    inline T Dot(const Vector<T, N>& v1, const Vector<T, N>& v2) {
        T r = static_cast<T>(0.0);

        for (int i = 0; i < N; i++)
            r += v1[i] * v2[i];

        return r;
    }

    // Calculate the cross product of vectors `v1` and `v2`
    template <typename T>
    inline Vector<T, 3> Cross(const Vector<T, 3>& v1, const Vector<T, 3>& v2) {
        return Vector<T, 3>{
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        };
    }

    /* Matrix functions */

    // Transpose matrix `m`
    template <typename T, int NRows, int NColumns>
    Matrix<T, NColumns, NRows> Transpose(const Matrix<T, NRows, NColumns>& m) {
        Matrix<T, NColumns, NRows> r;

        for (int i = 0; i < NRows; i++)
            for (int j = 0; j < NColumns; j++)
                r[j][i] = m[i][j];

        return r;
    }

    // Create left-handed perspective matrix
    // reference: https://github.com/g-truc/glm/blob/master/glm/ext/matrix_clip_space.inl#L265
    template <typename T>
    Matrix<T, 4, 4> PerspectiveLH(T fov, T aspect, T near, T far) {
        Matrix<T, 4, 4> r(static_cast<T>(0.0));

        const T tanHalfFOV = std::tan(fov / static_cast<T>(2.0));

        r[0][0] = static_cast<T>(1.0) / (aspect * tanHalfFOV);
        r[1][1] = static_cast<T>(1.0) / (tanHalfFOV);
        r[2][2] = (far + near) / (far - near);
        r[2][3] = -(static_cast<T>(2.0) * far * near) / (far - near);
        r[3][2] = static_cast<T>(1.0);

        return r;
    }

    // Translate matrix `m` by vector `v`
    template <typename T>
    Matrix<T, 4, 4> Translate(Matrix<T, 4, 4> m, const Vector<T, 3>& v) {
        // Dot product of each m Vector3 row and Vector3 of m's last column

        m[0][3] += m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
        m[1][3] += m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2];
        m[2][3] += m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2];

        return m;
    }

    // Rotate matrix `m` around `axis` by `angle` (radians)
    template <typename T>
    Matrix<T, 4, 4> Rotate(Matrix<T, 4, 4> m, T angle, const Vector<T, 3>& axis) {
        const T c = std::cos(angle);
        const T s = std::sin(angle);

        // Normalized inverse of C
        const T nic = static_cast<T>(1.0) - c;

        const T& x = axis[0];
        const T& y = axis[1];
        const T& z = axis[2];
        const T xx = x * x;
        const T yy = y * y;
        const T zz = z * z;
        const T xy = x * y;
        const T xz = x * z;
        const T yz = y * z;

        m[0] += {
            c + xx * nic,
            xy * nic + z * s,
            xz * nic - y * s,
            static_cast<T>(0.0)
        };

        m[1] += {
            xy * nic - z * s,
            c + yy * nic,
            yz * nic + x * s,
            static_cast<T>(0.0)
        };

        m[2] += {
            xz * nic + y * s,
            yz * nic - x * s,
            c + zz * nic,
            static_cast<T>(0.0)
        };

        return m;
    }

    // Scale matrix `m` by vector `v`
    template <typename T>
    Matrix<T, 4, 4> Scale(Matrix<T, 4, 4> m, const Vector<T, 3>& v) {
        m[0] *= v[0];
        m[1] *= v[1];
        m[2] *= v[2];
        
        return m;
    }

    // Create LookAt vector
    template <typename T>
    Matrix<T, 4, 4> LookAt(const Vector<T, 3>& eye, const Vector<T, 3>& target, const Vector<T, 3>& up) {
        const Vector<T, 3> eyeDir = Normalize(eye - target);
        const Vector<T, 3> eyeRight = Normalize(Cross(up, eyeDir));
        const Vector<T, 3> eyeUp = Cross(eyeDir, eyeRight);

        Matrix<T, 4, 4> r;
        r[0] = { eyeRight[0], eyeRight[1], eyeRight[2], static_cast<T>(0.0) };
        r[1] = { eyeUp[0], eyeUp[1], eyeUp[2], static_cast<T>(0.0) };
        r[2] = { eyeDir[0], eyeDir[1], eyeDir[2], static_cast<T>(0.0) };
        r[3] = { static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0) };
        

        return r * Translate(Matrix<T, 4, 4>(static_cast<T>(1.0)), eye);
    }

}