using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;

namespace DeepLearnCS
{
    public class ManagedLayerJSON
    {
        public int Type;
        public int OutputMaps;
        public int Scale;
        public int KernelSize;

        public double[,,,] FeatureMap; // FeatureMap[i][j][x][y]
        public double[] Bias;
    }

    public class ManagedCNNJSON
    {
        public List<ManagedLayerJSON> Layers = new List<ManagedLayerJSON>();

        public double[,] Weights;
        public double[] Bias;
    }

    public class ManagedNNJSON
    {
        public double[,] Wji;
        public double[,] Wkj;
    }

    public class ManagedDNNJSON
    {
        public List<double[,]> Weights = new List<double[,]>();
    }

    public static class Utility
    {
        public static double[] Convert1D(ManagedArray A)
        {
            var model = new double[A.Length()];

            for (var i = 0; i < A.Length(); i++)
                model[i] = A[i];

            return model;
        }

        public static double[,] Convert2D(ManagedArray A)
        {
            var model = new double[A.y, A.x];

            for (var y = 0; y < A.y; y++)
                for (var x = 0; x < A.x; x++)
                    model[y, x] = A[x, y];

            return model;
        }

        public static double[,,] Convert3D(ManagedArray A)
        {
            var model = new double[A.y, A.x, A.z];

            for (var z = 0; z < A.z; z++)
                for (var y = 0; y < A.y; y++)
                    for (var x = 0; x < A.x; x++)
                        model[y, x, z] = A[x, y, z];

            return model;
        }

        public static double[,,,] Convert4DIJ(ManagedArray A)
        {
            var model = new double[A.i, A.j, A.y, A.x];

            var temp = new ManagedArray(A.x, A.y);

            for (var i = 0; i < A.i; i++)
            {
                for (var j = 0; j < A.j; j++)
                {
                    ManagedOps.Copy4DIJ2D(temp, A, i, j);

                    for (var y = 0; y < A.y; y++)
                        for (var x = 0; x < A.x; x++)
                            model[i, j, y, x] = temp[x, y];
                }
            }

            ManagedOps.Free(temp);

            return model;
        }

        public static ManagedArray Set(double[] A, bool vert = false)
        {
            var ii = A.GetLength(0);

            var model = vert ? new ManagedArray(1, ii) : new ManagedArray(ii);

            for (var i = 0; i < ii; i++)
                model[i] = A[i];

            return model;
        }

        public static ManagedArray Set(double[,] A)
        {
            var yy = A.GetLength(0);
            var xx = A.GetLength(1);

            var model = new ManagedArray(xx, yy);

            for (var y = 0; y < yy; y++)
                for (var x = 0; x < xx; x++)
                    model[x, y] = A[y, x];

            return model;
        }

        public static ManagedArray Set(double[,,] A)
        {
            var yy = A.GetLength(0);
            var xx = A.GetLength(1);
            var zz = A.GetLength(2);

            var model = new ManagedArray(xx, yy, zz);

            for (var z = 0; z < zz; z++)
                for (var y = 0; y < yy; y++)
                    for (var x = 0; x < xx; x++)
                        model[x, y, z] = A[y, x, z];

            return model;
        }

        public static ManagedArray Set(double[,,,] A)
        {
            var ii = A.GetLength(0);
            var jj = A.GetLength(1);
            var yy = A.GetLength(2);
            var xx = A.GetLength(3);

            var model = new ManagedArray(xx, yy, 1, ii, jj);

            var temp = new ManagedArray(xx, yy);

            for (var i = 0; i < ii; i++)
            {
                for (var j = 0; j < jj; j++)
                {
                    for (var y = 0; y < yy; y++)
                        for (var x = 0; x < xx; x++)
                            temp[x, y] = A[i, j, y, x];

                    ManagedOps.Copy2D4DIJ(model, temp, i, j);
                }
            }

            ManagedOps.Free(temp);

            return model;
        }

        public static ManagedDNNJSON Convert(ManagedDNN network)
        {
            var model = new ManagedDNNJSON();

            for (var layer = 0; layer < network.Weights.Count; layer++)
            {
                model.Weights.Add(Convert2D(network.Weights[layer]));
            }

            return model;
        }

        public static string Serialize(ManagedDNN network)
        {
            var model = Convert(network);

            var output = JsonConvert.SerializeObject(model);

            return output;
        }

        public static ManagedDNN DeserializeDNN(string json)
        {
            var model = JsonConvert.DeserializeObject<ManagedDNNJSON>(json);

            var network = new ManagedDNN();

            for (var layer = 0; layer < model.Weights.Count; layer++)
            {
                network.Weights.Add(Set(model.Weights[layer]));
            }

            return network;
        }

        public static void Save(string BaseDirectory, string Filename, string json)
        {
            if (!string.IsNullOrEmpty(BaseDirectory) && !string.IsNullOrEmpty(Filename) && !string.IsNullOrEmpty(json))
            {
                var filename = string.Format("{0}/{1}.json", BaseDirectory, Filename);

                using (var file = new StreamWriter(filename, false))
                {
                    file.Write(json);
                }
            }
        }

        static string ReadString(string BaseDirectory, string Filename)
        {
            var json = "";

            if (!string.IsNullOrEmpty(BaseDirectory) && !string.IsNullOrEmpty(Filename))
            {
                var filename = string.Format("{0}/{1}.json", BaseDirectory, Filename);

                if (File.Exists(filename))
                {
                    using (var file = new StreamReader(filename))
                    {
                        var line = "";

                        while (!string.IsNullOrEmpty(line = file.ReadLine()))
                        {
                            json += line;
                        }
                    }
                }
            }

            return json;
        }

        public static ManagedDNN LoadDNN(string BaseDirectory, string Filename)
        {
            var json = ReadString(BaseDirectory, Filename);

            return !string.IsNullOrEmpty(json) ? DeserializeDNN(json) : null;
        }
    }
}
