using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;

namespace DeepLearnCS
{
    public class ManagedDNNJSON
    {
        public List<double[,]> Weights = new List<double[,]>();
        public List<double[]> Normalization = new List<double[]>();
    }

    public static class Utility
    {
        public static double[,] Convert2D(ManagedArray A)
        {
            var model = new double[A.y, A.x];

            for (var y = 0; y < A.y; y++)
                for (var x = 0; x < A.x; x++)
                    model[y, x] = A[x, y];

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

        public static ManagedDNNJSON Convert(ManagedDNN network)
        {
            var model = new ManagedDNNJSON();

            for (var layer = 0; layer < network.Weights.GetLength(0); layer++)
            {
                model.Weights.Add(Convert2D(network.Weights[layer]));
            }

            model.Normalization.Add(network.Min);
            model.Normalization.Add(network.Max);

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

            var network = new ManagedDNN()
            {
                Weights = new ManagedArray[model.Weights.Count],
                X = new ManagedArray[model.Weights.Count],
                Z = new ManagedArray[model.Weights.Count],
                D = new ManagedArray[model.Weights.Count],
                Deltas = new ManagedArray[model.Weights.Count],
                Activations = new ManagedArray[model.Weights.Count - 1],
                Layers = new List<HiddenLayer>()
            };

            for (var layer = 0; layer < model.Weights.Count; layer++)
            {
                network.Weights[layer] = Set(model.Weights[layer]);
                network.Layers.Add(new HiddenLayer(network.Weights[layer].x - 1, network.Weights[layer].y));
            }

            if (model.Normalization != null && model.Normalization.Count > 1)
            {
                network.Min = model.Normalization[0];
                network.Max = model.Normalization[1];
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

        static string LoadJSON(string BaseDirectory, string Filename)
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

        public static ManagedDNN LoadDNN(string BaseDirectory, string Filename, ManagedArray normalization)
        {
            var json = LoadJSON(BaseDirectory, Filename);

            var network = !string.IsNullOrEmpty(json) ? DeserializeDNN(json) : null;

            if (network.Min.GetLength(0) > 0 && network.Max.GetLength(0) > 0)
            {
                if (normalization == null)
                {
                    normalization = new ManagedArray(network.Min.GetLength(0), 2);
                }
                else
                {
                    normalization.Resize(network.Min.GetLength(0), 2);
                }

                for (var x = 0; x < 2; x++)
                {
                    normalization[x, 0] = network.Min[x];
                    normalization[x, 1] = network.Max[x];
                }
            }

            return network;
        }
    }
}
