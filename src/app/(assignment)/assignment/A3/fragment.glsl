uniform float iTime;
uniform vec2 iResolution;

//======================================================= Copy-Paste Area Begin =========================================================

// Number of Gaussians
const int NUM_GAUSSIANS = 100;
// Dimensions [x_min, x_max, y_min, y_max]
float dim[4] = float[4](-5.0,5.0,-6.468253968253968,6.468253968253968);
// Centers (x, y coordinates)
vec2 gauss_centers[NUM_GAUSSIANS] = vec2[NUM_GAUSSIANS](vec2(0.56, -1.47),vec2(-1.02, 2.17),vec2(-2.90, -1.46),vec2(-2.93, 1.09),vec2(4.81, 5.52),vec2(-5.57, -5.31),vec2(-0.92, -2.56),vec2(2.02, -6.21),vec2(1.12, 1.74),vec2(6.32, -4.97),vec2(-4.01, 4.72),vec2(-3.78, -2.83),vec2(2.57, 3.32),vec2(3.58, 0.71),vec2(-2.24, 7.15),vec2(-2.21, -5.10),vec2(2.82, 3.05),vec2(2.26, 3.90),vec2(-3.75, -1.78),vec2(-5.69, -3.24),vec2(0.74, 3.70),vec2(-4.53, -1.47),vec2(-2.38, 6.44),vec2(4.39, 3.51),vec2(-3.10, 1.70),vec2(-5.20, 6.04),vec2(-0.51, -4.91),vec2(1.62, -6.86),vec2(3.38, -3.51),vec2(-2.50, 6.70),vec2(4.63, -2.00),vec2(3.36, 1.85),vec2(-3.58, 3.47),vec2(-5.66, -2.93),vec2(-2.30, -1.66),vec2(-3.55, -3.33),vec2(0.43, -2.56),vec2(-2.44, 0.04),vec2(-4.78, 8.02),vec2(-1.71, 1.42),vec2(0.48, 5.96),vec2(-1.45, 0.29),vec2(-1.33, -1.47),vec2(4.15, 0.37),vec2(-3.16, -3.52),vec2(2.47, 1.34),vec2(2.11, 3.91),vec2(-3.38, -5.06),vec2(-4.16, -5.53),vec2(0.59, 5.62),vec2(-2.03, -3.60),vec2(4.40, 0.86),vec2(-1.96, 4.19),vec2(1.82, -2.61),vec2(-4.12, -4.46),vec2(2.43, 3.88),vec2(3.04, 5.67),vec2(0.29, -3.24),vec2(-3.07, 1.28),vec2(1.16, 3.19),vec2(-0.66, -6.60),vec2(-3.79, 3.46),vec2(1.02, -0.00),vec2(-0.51, 5.34),vec2(3.41, -0.15),vec2(-2.16, -4.94),vec2(3.93, -5.13),vec2(-0.60, -0.78),vec2(-2.69, 3.17),vec2(-3.03, -0.50),vec2(-3.47, 5.40),vec2(2.92, 7.13),vec2(2.95, 6.12),vec2(2.76, -6.15),vec2(-4.23, 2.91),vec2(5.04, 7.27),vec2(-1.97, 2.41),vec2(0.38, -2.89),vec2(1.07, -3.44),vec2(-3.25, 3.80),vec2(1.55, -0.79),vec2(-1.96, -1.41),vec2(4.60, -4.35),vec2(-1.58, 7.04),vec2(-1.85, -3.64),vec2(-3.48, -2.75),vec2(-3.98, -0.12),vec2(-3.66, -4.02),vec2(-1.27, 4.25),vec2(-2.76, -5.51),vec2(-0.06, 5.27),vec2(3.24, -1.21),vec2(3.77, 0.42),vec2(4.20, -3.10),vec2(2.85, 2.65),vec2(0.66, -0.61),vec2(0.92, 1.12),vec2(-0.53, -5.32),vec2(4.47, 3.60),vec2(-2.95, -3.58));
// Sigmas (scales)
vec2 gauss_sigmas[NUM_GAUSSIANS] = vec2[NUM_GAUSSIANS](vec2(0.48, 1.19),vec2(0.72, 0.97),vec2(0.32, 0.68),vec2(0.45, 1.01),vec2(-1.06, -0.84),vec2(-0.88, -0.24),vec2(0.37, 1.44),vec2(1.83, 0.24),vec2(0.67, 1.10),vec2(-0.22, -0.36),vec2(-0.43, -0.22),vec2(0.18, 0.35),vec2(-0.02, -0.03),vec2(0.38, 0.79),vec2(0.47, 1.19),vec2(1.47, 1.17),vec2(1.15, -0.13),vec2(-0.00, 0.05),vec2(0.22, 0.94),vec2(-0.37, -0.27),vec2(-0.06, 0.07),vec2(1.07, 1.56),vec2(1.33, 0.42),vec2(-0.64, -0.36),vec2(1.00, 0.32),vec2(1.18, 1.38),vec2(0.55, 0.11),vec2(0.04, 0.05),vec2(1.55, 0.26),vec2(-0.97, -0.29),vec2(0.74, 0.22),vec2(0.31, 0.53),vec2(-1.14, -0.24),vec2(0.25, 0.50),vec2(0.58, 0.23),vec2(0.63, 0.62),vec2(0.52, 0.76),vec2(1.72, 0.38),vec2(1.11, 0.32),vec2(0.83, 0.24),vec2(-0.60, -0.14),vec2(1.05, 0.72),vec2(0.44, 0.70),vec2(0.35, 1.21),vec2(0.10, 0.66),vec2(0.00, 0.00),vec2(-0.26, -0.46),vec2(-0.29, -0.18),vec2(1.01, 0.49),vec2(0.09, 0.18),vec2(0.28, 1.13),vec2(-0.02, 0.09),vec2(0.14, 1.02),vec2(1.08, 0.71),vec2(-0.21, -0.07),vec2(-0.57, -0.86),vec2(-0.69, -0.87),vec2(0.69, 0.37),vec2(1.13, 0.51),vec2(-0.56, -0.13),vec2(-1.38, -0.19),vec2(0.91, 1.57),vec2(0.63, 0.74),vec2(1.26, 1.95),vec2(0.22, 0.39),vec2(-0.45, -0.09),vec2(0.00, -0.01),vec2(0.27, 0.66),vec2(0.30, 0.24),vec2(0.36, 0.24),vec2(-0.26, -0.64),vec2(-0.16, 0.01),vec2(-0.16, -0.38),vec2(0.70, 1.48),vec2(0.26, 1.23),vec2(-0.30, -0.17),vec2(0.65, 0.27),vec2(0.99, 0.44),vec2(0.01, -0.00),vec2(0.38, 0.24),vec2(0.42, 0.27),vec2(0.95, 0.85),vec2(0.36, 1.00),vec2(-0.14, 0.10),vec2(0.32, 0.58),vec2(-0.14, -0.08),vec2(0.27, 2.43),vec2(0.49, 0.20),vec2(0.43, 0.15),vec2(0.03, -0.08),vec2(0.05, -0.14),vec2(0.80, 0.49),vec2(-0.13, -0.07),vec2(0.13, 0.51),vec2(0.38, 2.49),vec2(1.21, 0.18),vec2(0.24, -1.05),vec2(-0.09, 0.05),vec2(-0.13, -0.07),vec2(1.14, 0.16));
// Rotation angles (thetas)
float gauss_thetas[NUM_GAUSSIANS] = float[NUM_GAUSSIANS](0.44,1.03,-0.44,0.40,-1.30,0.91,-0.69,0.29,-0.30,0.29,-0.72,0.17,-0.06,-1.30,-0.28,-1.02,-0.13,1.60,0.27,-0.91,-1.54,0.04,-0.34,1.75,0.74,2.67,0.59,-0.14,1.23,2.60,1.43,-1.44,0.81,-1.36,-0.67,-0.45,-1.08,-0.14,0.21,-0.47,0.90,-0.68,-0.96,0.32,-0.98,0.63,-0.68,0.09,0.99,1.89,-1.68,-1.88,-1.57,-0.53,1.14,-2.05,1.65,-0.71,-0.20,0.46,0.35,1.88,-0.40,-0.04,0.07,0.48,-0.99,-0.60,-0.67,-1.31,0.80,0.15,-1.44,0.92,-0.07,-0.29,-0.71,0.15,1.15,0.02,0.91,0.83,-0.44,1.57,-0.82,-1.46,0.71,-0.91,-0.06,1.35,-0.72,-0.30,-2.30,-0.15,1.30,-0.06,0.44,-1.03,-2.70,0.79);
// Colors (RGB)
vec3 gauss_colors[NUM_GAUSSIANS] = vec3[NUM_GAUSSIANS](vec3(0.45, 0.44, 0.39),vec3(0.22, 0.33, 0.31),vec3(0.22, 0.19, 0.21),vec3(0.29, 0.32, 0.35),vec3(0.04, 0.05, 0.04),vec3(0.14, 0.14, 0.10),vec3(0.61, 0.50, 0.44),vec3(0.33, 0.35, 0.28),vec3(0.19, 0.19, 0.14),vec3(-0.36, -0.01, -0.55),vec3(0.05, 0.10, 0.15),vec3(0.31, 0.44, 0.50),vec3(-0.04, -0.05, -0.04),vec3(0.43, 0.50, 0.43),vec3(0.37, 0.46, 0.44),vec3(0.17, 0.22, 0.19),vec3(0.08, 0.08, 0.06),vec3(0.03, -0.04, -0.01),vec3(0.20, 0.37, 0.43),vec3(0.37, -0.09, -0.19),vec3(-0.02, -0.04, -0.07),vec3(0.25, 0.21, 0.14),vec3(0.26, 0.36, 0.29),vec3(0.09, 0.09, 0.06),vec3(0.26, 0.26, 0.25),vec3(0.18, 0.25, 0.23),vec3(0.22, 0.23, 0.21),vec3(0.30, 0.09, -0.10),vec3(0.09, 0.09, 0.09),vec3(-0.26, -0.34, -0.27),vec3(0.41, 0.44, 0.34),vec3(0.25, 0.35, 0.32),vec3(0.27, 0.29, 0.37),vec3(0.54, 0.67, 0.59),vec3(-0.20, -0.17, -0.10),vec3(-0.15, -0.20, -0.22),vec3(0.41, -0.09, -0.17),vec3(0.13, 0.14, 0.08),vec3(0.10, 0.65, 0.08),vec3(0.25, 0.30, 0.30),vec3(-0.02, -0.03, -0.02),vec3(0.29, 0.41, 0.47),vec3(0.66, 0.34, 0.24),vec3(0.14, 0.16, 0.12),vec3(-0.22, -0.25, -0.24),vec3(0.03, -0.07, -0.05),vec3(-0.04, -0.04, -0.04),vec3(-0.12, -0.15, -0.14),vec3(0.07, 0.08, 0.06),vec3(-0.03, -0.04, -0.02),vec3(0.26, 0.33, 0.32),vec3(-0.12, -0.13, -0.14),vec3(-0.18, -0.24, -0.27),vec3(0.19, 0.24, 0.22),vec3(0.22, 0.22, 0.19),vec3(0.07, 0.07, 0.06),vec3(0.07, 0.08, 0.07),vec3(0.31, 0.41, 0.38),vec3(0.09, 0.19, 0.16),vec3(-0.06, -0.07, -0.06),vec3(0.11, 0.24, 0.26),vec3(0.37, 0.48, 0.39),vec3(0.16, 0.28, 0.27),vec3(0.12, 0.14, 0.11),vec3(0.14, 0.25, 0.24),vec3(-0.14, -0.15, -0.12),vec3(-0.11, -0.06, -0.12),vec3(0.16, 0.24, 0.25),vec3(0.31, 0.35, 0.40),vec3(0.31, 0.36, 0.40),vec3(0.10, 0.15, 0.15),vec3(-0.27, -0.39, -0.41),vec3(-0.03, -0.04, -0.03),vec3(0.20, 0.20, 0.15),vec3(0.09, 0.11, 0.11),vec3(-0.07, -0.29, -0.32),vec3(0.27, 0.31, 0.31),vec3(-0.37, -0.04, 0.03),vec3(-0.04, -0.03, -0.09),vec3(-0.23, -0.30, -0.45),vec3(0.52, 0.52, 0.48),vec3(0.18, 0.40, 0.43),vec3(0.48, 0.48, 0.34),vec3(-0.24, -0.20, 0.02),vec3(-0.61, -0.61, -0.55),vec3(0.16, 0.13, 0.12),vec3(0.10, 0.15, 0.13),vec3(0.42, 0.42, 0.44),vec3(0.24, 0.33, 0.35),vec3(-0.12, -0.12, -0.13),vec3(0.02, 0.04, 0.03),vec3(0.18, 0.30, 0.32),vec3(-0.28, -0.32, -0.27),vec3(0.42, 0.50, 0.43),vec3(0.09, 0.11, 0.09),vec3(-0.08, -0.14, -0.15),vec3(-0.05, -0.08, -0.08),vec3(0.23, 0.28, 0.26),vec3(0.18, 0.19, 0.14),vec3(0.17, 0.17, 0.17));


//======================================================= Copy-Paste Area End =========================================================

/////////////////////////////////////////////////////
//// Here, you are asked to build the inverse covariance matrix, similar to in the 2DGS_A3_solution.ipynb file.
//// You must create the rotation matrix R, the inverse squared sigma matrix D, and the final inverse covariance matrix. 
/////////////////////////////////////////////////////

// This function builds the inverse covariance matrix
mat2 buildSigmaInv(float theta, vec2 sigma)
{
    mat2 cov_mat = mat2(0, 0, 0, 0);

    /////////// 
    // BEGINNING OF YOUR CODE.
    //////////

    // Rotation matrix R
    mat2 R = mat2(cos(theta), sin(theta),
                    -sin(theta), cos(theta));

    const float EPSILON = 1e-4;
    vec2 safeSigma = max(sigma, vec2(EPSILON));

    mat2 D = mat2(1.0 / (safeSigma.x * safeSigma.x), 0.0,
                  0.0, 1.0 / (safeSigma.y * safeSigma.y));

    cov_mat = R * D * transpose(R);

    /////////// 
    // END OF YOUR CODE.
    //////////
    return cov_mat;
}

/////////////////////////////////////////////////////
//// Here, you are asked to fill in the necessary components for calculating each gaussian's contribution to the current pixel's color.
//// You must calculate the position of the pixel relative to the gaussian's center, calculate the contribution exponent (pos^T * sigma_inv * pos) 
//// and finally calculate the Gaussian function value that will control the contribution of this specific gaussian.
/////////////////////////////////////////////////////

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    float aspect = (dim[1] - dim[0]) / (dim[3] - dim[2]) * iResolution.y / iResolution.x;
    vec2 uv = fragCoord.xy / iResolution.xy; // scale to [0, 1]
    // scale from [-1, 1] to [x_min, x_max] and [y_min, y_max]
    uv.x = mix(dim[0], dim[1], uv.x);
    uv.y = mix(dim[2], dim[3], uv.y);
    if (aspect > 1.0) {
        uv.y *= aspect;
    } else {
        uv.x /= aspect;
    }

    vec3 color = vec3(0.0);

    // Draw bounding box
    float edge = 0.01;
    if (uv.x > dim[0] - edge && uv.x < dim[0] + edge ||
        uv.x > dim[1] - edge && uv.x < dim[1] + edge ||
        ((uv.y > dim[2] - edge && uv.y < dim[2] + edge || uv.y > dim[3] - edge && uv.y < dim[3] + edge) && uv.x > dim[0] && uv.x < dim[1])) {
        color = vec3(1.0, 1.0, 1.0);
    }

    // Animate the Gaussian centers
    uv += smoothstep(0.0, 1.0, cos(iTime)) * cos(iTime + 100.0 * uv.xy) * 3.;

    for (int i = 0; i < NUM_GAUSSIANS; ++i) {
        vec2 center = gauss_centers[i];
        vec2 scale = gauss_sigmas[i];
        float theta = gauss_thetas[i];
        vec3 color_rgb = gauss_colors[i];
        
        // Build inverse covariance matrix
        mat2 sigma_inv = buildSigmaInv(theta, scale);
        float f_x = 0.;

        /////////// 
        // BEGINNING OF YOUR CODE.
        //////////

        vec2 pos = uv - center;

        float exponent = dot(pos, sigma_inv * pos);

        f_x = exp(-0.5 * exponent);

        /////////// 
        // END OF YOUR CODE.
        //////////
        
        // Add color contribution
        color += f_x * color_rgb;
    }

    fragColor = vec4(color, 1.0);
}


void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}