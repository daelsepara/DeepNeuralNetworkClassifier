using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;

namespace DeepLearnCS
{
    public class ManagedDNNJSON
    {
        public List<double[,]> Weights = new List<double[,]>();
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
