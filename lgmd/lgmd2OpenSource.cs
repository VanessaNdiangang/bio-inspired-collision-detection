using System;

namespace LGMD
{
    /// <summary>
    /// Implements the LGMD2 visual neural network inspired by biological collision detection neurons.
    /// Selectively detects dark looming stimuli using ON/OFF pathways and adaptive inhibition mechanisms.
    /// </summary>
    public class LGMD2OpenSource
    {
        #region Fields

        // Input image/frame dimensions
        protected readonly int width;
        protected readonly int height;
        protected readonly int Ncell;

        // Convolution kernel radius
        protected readonly int Np;

        // Spike-frequency adaptation
        protected int Nsp;
        protected readonly int Nts;
        protected readonly int tau_sfa;
        protected readonly float hp_sfa;

        // Neural thresholds and weights
        protected readonly int Tpm;
        protected readonly int Cspi;
        protected readonly float clip_point;
        protected readonly float W_on, W_off, W_onoff;
        protected readonly float Tsp, Tsfa;
        protected float W_i_on, W_i_on_base;
        protected float W_i_off, W_i_off_base;
        protected readonly float dc;
        protected readonly float Csig;

        // Convolution kernels
        protected float[,] Conv_ON;
        protected float[,] Conv_OFF;
        protected float[,] W_g;

        // Lateral inhibition layers
        protected float[,] Inh_ON;
        protected float[,] Inh_OFF;

        // Temporal filtering parameters
        protected readonly float[] tau_ON;
        protected readonly float[] tau_OFF;
        protected readonly float tau_PM;
        protected readonly float[] lp_ON;
        protected readonly float[] lp_OFF;
        protected float lp_PM;

        // Grouping parameters
        protected readonly int Cw;
        protected readonly float Delta_C;
        protected readonly float Cde;
        protected readonly int Tde;

        // Time between video frames (ms)
        protected float time_interval;

        // Neural layers
        protected int[,,] photoreceptors;
        protected float[,,] ons;
        protected float[,,] offs;
        protected float[,] scells;
        protected float[,] gcells;

        // Intermediate state
        protected float S_on, S_off;
        protected float[] pm;
        protected float[] mp;
        protected float[] smp;
        protected float[] sfa;

        // Outputs
        protected byte spike;
        protected byte collision;

        // Exposed public properties
        public byte Spike => spike;
        public byte Collision => collision;
        public float MembranePotential => mp[1];
        public int TotalSpikeCount => Nsp;
        public float MotionEnergy => pm[1];
        public float SFA => sfa[1];

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes the LGMD2 system with frame resolution and frame rate.
        /// </summary>
        public LGMD2OpenSource(int width, int height, int fps)
        {
            this.width = width;
            this.height = height;
            this.time_interval = 1000f / fps;

            Ncell = width * height;
            Np = 1;
            Nts = 8;
            tau_sfa = 800;
            hp_sfa = tau_sfa / (tau_sfa + time_interval);

            Tpm = 8;
            Cspi = 4;
            clip_point = 0.1f;
            Tsp = 0.78f;
            Tsfa = 0.003f;
            dc = 0.1f;
            Csig = 1f;
            W_on = W_off = W_onoff = 1f;
            W_i_on = W_i_on_base = 1f;
            W_i_off = W_i_off_base = 0.5f;

            Cw = 4;
            Delta_C = 0.01f;
            Cde = 0.5f;
            Tde = 15;

            photoreceptors = new int[height, width, 2];
            ons = new float[height, width, 2];
            offs = new float[height, width, 2];
            scells = new float[height, width];
            gcells = new float[height, width];
            pm = new float[2];
            mp = new float[2];
            smp = new float[2];
            sfa = new float[2];

            spike = 0;
            collision = 0;

            Conv_ON = makeONConvKernel();
            Conv_OFF = makeOFFConvKernel();
            W_g = new float[2 * Np + 1, 2 * Np + 1];
            groupKernel(ref W_g);

            Inh_ON = new float[height, width];
            Inh_OFF = new float[height, width];

            tau_ON = new float[2 * Np + 1];
            tau_OFF = new float[2 * Np + 1];
            lp_ON = new float[2 * Np + 1];
            lp_OFF = new float[2 * Np + 1];

            for (int i = 0; i < 2 * Np + 1; i++)
            {
                tau_ON[i] = 15 + i * 15;
                tau_OFF[i] = 60 + i * 60;
                lp_ON[i] = time_interval / (time_interval + tau_ON[i]);
                lp_OFF[i] = time_interval / (time_interval + tau_OFF[i]);
            }

            tau_PM = 90;
            lp_PM = time_interval / (time_interval + tau_PM);
        }

        #endregion
        #region Kernel & Filtering Methods

        /// <summary>
        /// Generates the ON convolution kernel.
        /// </summary>
        protected float[,] makeONConvKernel()
        {
            var mat = new float[2 * Np + 1, 2 * Np + 1];
            for (int i = -1; i <= Np; i++)
            {
                for (int j = -1; j <= Np; j++)
                {
                    if (i == 0 && j == 0)
                        mat[i + 1, j + 1] = 2f;
                    else if (i == 0 || j == 0)
                        mat[i + 1, j + 1] = 0.5f;
                    else
                        mat[i + 1, j + 1] = 0.25f;
                }
            }
            return mat;
        }

        /// <summary>
        /// Generates the OFF convolution kernel.
        /// </summary>
        protected float[,] makeOFFConvKernel()
        {
            var mat = new float[2 * Np + 1, 2 * Np + 1];
            for (int i = -1; i <= Np; i++)
            {
                for (int j = -1; j <= Np; j++)
                {
                    if (i == 0 && j == 0)
                        mat[i + 1, j + 1] = 1f;
                    else if (i == 0 || j == 0)
                        mat[i + 1, j + 1] = 0.25f;
                    else
                        mat[i + 1, j + 1] = 0.125f;
                }
            }
            return mat;
        }

        /// <summary>
        /// Initializes the grouping layer convolution kernel.
        /// </summary>
        protected void groupKernel(ref float[,] mat)
        {
            for (int i = 0; i < 2 * Np + 1; i++)
            {
                for (int j = 0; j < 2 * Np + 1; j++)
                {
                    mat[i, j] = 1f / 9f;
                }
            }
        }

        /// <summary>
        /// First-order high-pass filter for frame difference.
        /// </summary>
        protected int HighpassFilter(byte pre_input, byte cur_input)
        {
            return cur_input - pre_input;
        }

        /// <summary>
        /// First-order low-pass filter.
        /// </summary>
        protected float LowpassFilter(float cur_input, float pre_input, float lp_t)
        {
            return lp_t * cur_input + (1 - lp_t) * pre_input;
        }

        #endregion

        #region ON/OFF Convolution and Grouping

        /// <summary>
        /// Applies spatial-temporal convolution with low-pass filtering.
        /// </summary>
        protected float Convolution(int x, int y, float[,,] inputMatrix, float[,] kernel, int cur_frame, int pre_frame, float[] lp_delay)
        {
            float result = 0;
            int r, c;
            for (int i = -Np; i <= Np; i++)
            {
                r = Math.Clamp(x + i, 0, height - 1);
                for (int j = -Np; j <= Np; j++)
                {
                    c = Math.Clamp(y + j, 0, width - 1);
                    float lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[Math.Abs(i) + Math.Abs(j)]);
                    result += lp * kernel[i + Np, j + Np];
                }
            }
            return result;
        }

        /// <summary>
        /// Convolution for grouping layer.
        /// </summary>
        protected float Convolving(int x, int y, float[,] matrix, float[,] kernel)
        {
            float result = 0;
            int r, c;
            for (int i = -Np; i <= Np; i++)
            {
                r = Math.Clamp(x + i, 0, height - 1);
                for (int j = -Np; j <= Np; j++)
                {
                    c = Math.Clamp(y + j, 0, width - 1);
                    result += matrix[r, c] * kernel[i + Np, j + Np];
                }
            }
            return result;
        }

        /// <summary>
        /// Computes the scale value for grouping layer normalization.
        /// </summary>
        protected float Scale()
        {
            float max = 0;
            foreach (var value in gcells)
                if (Math.Abs(value) > max)
                    max = Math.Abs(value);

            return Delta_C + max / Cw;
        }

        #endregion

        #region Signal Processing Methods

        protected float HRplusDC_ON(float pre_output, float cur_input)
        {
            return cur_input >= clip_point ? cur_input + dc * pre_output : dc * pre_output;
        }

        protected float HRplusDC_OFF(float pre_output, float cur_input)
        {
            return cur_input < clip_point ? Math.Abs(cur_input) + dc * pre_output : dc * pre_output;
        }

        protected float sCellValue(float exc, float inh, float wi)
        {
            float result = exc - inh * wi;
            return result > 0 ? result : 0;
        }

        protected float SupralinearSummation(float on_exc, float off_exc)
        {
            return W_on * on_exc + W_off * off_exc + W_onoff * on_exc * off_exc;
        }

        protected float gCellValue(float scellvalue, float ce, float w)
        {
            float value = scellvalue * ce / w;
            return value * Cde >= Tde ? value : 0;
        }

        protected float SigmoidTransfer(float Kf)
        {
            return (float)(1.0 / (1 + Math.Exp(-Kf / (Ncell * Csig))));
        }

        protected float SFA_HPF(float pre_sfa, float pre_mp, float cur_mp)
        {
            float diff_mp = cur_mp - pre_mp;
            float tmp_mp = diff_mp <= Tsfa ? hp_sfa * (pre_sfa + diff_mp) : hp_sfa * cur_mp;
            return Math.Max(tmp_mp, 0.5f);
        }

        protected byte Spiking(float sfa)
        {
            byte spikes = (byte)Math.Floor(Math.Exp(Cspi * (sfa - Tsp)));
            Nsp = spikes == 0 ? 0 : Nsp + spikes;
            return spikes;
        }

        protected byte loomingDetecting()
        {
            return (byte)(Nsp >= Nts ? 1 : 0);
        }

        #endregion
        #region Main LGMD2 Processing Pipeline

        /// <summary>
        /// Main LGMD2 visual processing method for one time-step.
        /// </summary>
        public void LGMD2_Processing(byte[,,] img1, byte[,,] img2, int t)
        {
            int cur_frame = t % 2;
            int pre_frame = (t - 1) % 2;
            float tmp_pm = 0;
            float tmp_sum = 0;

            // Photoreceptor layer (P-layer)
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    photoreceptors[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);
                    tmp_pm += Math.Abs(photoreceptors[y, x, cur_frame]);
                }
            }

            // Photoreceptor mediation adaptation
            pm[cur_frame] = tmp_pm / Ncell;
            pm[cur_frame] = LowpassFilter(pm[cur_frame], pm[pre_frame], lp_PM);

            W_i_off = Math.Max(pm[cur_frame] / Tpm, W_i_off_base);
            W_i_on = Math.Max(pm[cur_frame] / Tpm, W_i_on_base);

            // ON/OFF Rectification
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ons[y, x, cur_frame] = HRplusDC_ON(ons[y, x, pre_frame], photoreceptors[y, x, cur_frame]);
                    offs[y, x, cur_frame] = HRplusDC_OFF(offs[y, x, pre_frame], photoreceptors[y, x, cur_frame]);
                }
            }

            // Summation layer
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Inh_ON[y, x] = Convolution(y, x, ons, Conv_ON, cur_frame, pre_frame, lp_ON);
                    Inh_OFF[y, x] = Convolution(y, x, offs, Conv_OFF, cur_frame, pre_frame, lp_OFF);
                    S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], W_i_on);
                    S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], W_i_off);
                    scells[y, x] = SupralinearSummation(S_on, S_off);
                }
            }

            // Grouping layer
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    gcells[y, x] = Convolving(y, x, scells, W_g);

            float scale = Scale();

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x] = gCellValue(scells[y, x], gcells[y, x], scale);
                    tmp_sum += gcells[y, x];
                }

            // LGMD neuron integration
            mp[cur_frame] = tmp_sum;
            smp[cur_frame] = SigmoidTransfer(mp[cur_frame]);
            sfa[cur_frame] = SFA_HPF(sfa[pre_frame], smp[pre_frame], smp[cur_frame]);

            spike = Spiking(sfa[cur_frame]);
            collision = loomingDetecting();
        }

        #endregion
    }
}
