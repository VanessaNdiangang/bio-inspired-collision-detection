using System;
using System.IO;
using OpenCvSharp;
using LGMD;
using System.Globalization;

namespace ConsoleProject
{
    class Program
    {
        /// <summary>
        /// Entry point for the LGMD model.
        /// Usage: ConsoleProject.exe [inputVideoPath] [optionalOutputCsvPath]
        /// </summary>
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.Error.WriteLine("[ERROR] No input video file provided.");
                return;
            }

            string videoPath = args[0];
            if (!File.Exists(videoPath))
            {
                Console.Error.WriteLine($"[ERROR] Video file not found: {videoPath}");
                return;
            }

            // Optional output file path; default to local .csv
            string baseName = Path.GetFileNameWithoutExtension(videoPath);
            string logPath = args.Length > 1 ? args[1] : $"{baseName}_lgmd_output.csv";

            Console.WriteLine($"\n[INFO] Processing video: {baseName}");

            try
            {
                using var cap = new VideoCapture(videoPath);
                if (!cap.IsOpened())
                {
                    Console.Error.WriteLine($"[ERROR] Failed to open video: {videoPath}");
                    return;
                }

                int width = cap.FrameWidth;
                int height = cap.FrameHeight;
                int fps = (int)Math.Round(cap.Fps);

                var lgmd = new LGMD2OpenSource(width, height, fps);

                using var mat0 = new Mat();
                if (!cap.Read(mat0))
                {
                    Console.Error.WriteLine("[ERROR] Could not read the first frame.");
                    return;
                }

                var prevGray = new Mat();
                Cv2.CvtColor(mat0, prevGray, ColorConversionCodes.BGR2GRAY);
                var prevFrame = ToByteArray(prevGray);

                int frameIdx = 1;
                using var mat = new Mat();

                using var logWriter = new StreamWriter(logPath);
                logWriter.WriteLine("Frame,MembranePotential,Spikes,Collision,TotalSpikes,MotionEnergy,SFA");

                int collisionCount = 0;
                float maxMembranePotential = float.MinValue;

                while (cap.Read(mat))
                {
                    var curGray = new Mat();
                    Cv2.CvtColor(mat, curGray, ColorConversionCodes.BGR2GRAY);
                    var curFrame = ToByteArray(curGray);

                    lgmd.LGMD2_Processing(prevFrame, curFrame, frameIdx);

                    if (lgmd.Collision == 1)
                    {
                        double tSec = frameIdx / (double)fps;
                        Console.WriteLine($"  Collision detected at frame {frameIdx} (~{tSec:F2}s), spikes = {lgmd.Spike}");
                        collisionCount++;
                    }

                    if (lgmd.MembranePotential > maxMembranePotential)
                        maxMembranePotential = lgmd.MembranePotential;

                    logWriter.WriteLine($"{frameIdx.ToString(CultureInfo.InvariantCulture)}," +
                        $"{lgmd.MembranePotential.ToString("F4", CultureInfo.InvariantCulture)}," +
                        $"{lgmd.Spike},{lgmd.Collision}," +
                        $"{lgmd.TotalSpikeCount}," +
                        $"{lgmd.MotionEnergy.ToString("F4", CultureInfo.InvariantCulture)}," +
                        $"{lgmd.SFA.ToString("F4", CultureInfo.InvariantCulture)}");

                    prevFrame = curFrame;
                    frameIdx++;
                }

                Console.WriteLine($"[DONE] {baseName}: {frameIdx} frames, {collisionCount} collisions, max potential {maxMembranePotential:F4}");
                Console.WriteLine($"[OUTPUT] CSV saved to: {logPath}");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ERROR] Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Convert grayscale Mat to 3D byte array for model input
        /// </summary>
        static byte[,,] ToByteArray(Mat gray)
        {
            int H = gray.Rows, W = gray.Cols;
            var arr = new byte[H, W, 1];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    arr[y, x, 0] = gray.At<byte>(y, x);
            return arr;
        }
    }
}
