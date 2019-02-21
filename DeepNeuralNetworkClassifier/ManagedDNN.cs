using System;
using System.Collections.Generic;

namespace DeepLearnCS
{
    public class ManagedDNN
    {
        public List<ManagedArray> Weights = new List<ManagedArray>();
        public List<ManagedArray> Deltas = new List<ManagedArray>();

        public ManagedArray Yk;
        public ManagedArray Y_output;

        // intermediate results
        public List<ManagedArray> X = new List<ManagedArray>();
        public List<ManagedArray> Z = new List<ManagedArray>();

        // Error
        public double Cost;
        public double L2;

        public int Iterations;

        // Forward Propagation
        public void Forward(ManagedArray input)
        {
            var Activations = new List<ManagedArray>();

            // create bias column
            var InputBias = new ManagedArray(1, input.y);
            ManagedOps.Set(InputBias, 1.0);

            // Compute input activations
            var last = Weights.Count - 1;

            for (var layer = 0; layer < Weights.Count; layer++)
            {
                var XX = layer == 0 ? ManagedMatrix.CBind(InputBias, input) : ManagedMatrix.CBind(InputBias, Activations[layer - 1]);
                var tW = ManagedMatrix.Transpose(Weights[layer]);
                var ZZ = ManagedMatrix.Multiply(XX, tW);

                X.Add(XX);
                Z.Add(ZZ);

                if (layer != last)
                {
                    var SS = ManagedMatrix.Sigm(ZZ);

                    Activations.Add(SS);
                }
                else
                {
                    Yk = ManagedMatrix.Sigm(ZZ);
                }

                ManagedOps.Free(tW);
            }

            // Cleanup
            for (var layer = 0; layer < Activations.Count; layer++)
            {
                ManagedOps.Free(Activations[layer]);
            }

            Activations.Clear();

            ManagedOps.Free(InputBias);
        }

        // Backward propagation
        public void BackPropagation(ManagedArray input)
        {
            var D = new List<ManagedArray>();

            var last = Weights.Count - 1;

            D.Add(ManagedMatrix.Diff(Yk, Y_output));

            for (var layer = last - 1; layer >= 0; layer--)
            {
                var current = D.Count;
                var prev = D.Count - 1;
                
                var W = new ManagedArray(Weights[layer + 1].x - 1, Weights[layer + 1].y);
                var DZ = ManagedMatrix.DSigm(Z[layer]);

                D.Add(new ManagedArray(W.x, D[prev].y));

                ManagedOps.Copy2D(W, Weights[layer + 1], 1, 0);
                ManagedMatrix.Multiply(D[current], D[prev], W);
                ManagedMatrix.Product(D[current], DZ);

                ManagedOps.Free(W, DZ);
            }

            for (var layer = 0; layer < Weights.Count; layer++)
            {
                var tD = ManagedMatrix.Transpose(D[Weights.Count - layer - 1]);

                Deltas.Add(new ManagedArray(Weights[layer].x, Weights[layer].y));

                ManagedMatrix.Multiply(Deltas[layer], tD, X[layer]);
                ManagedMatrix.Multiply(Deltas[layer], 1.0 / input.y);

                ManagedOps.Free(tD);
            }

            Cost = 0.0;
            L2 = 0.0;

            for (var i = 0; i < Y_output.Length(); i++)
            {
                L2 += 0.5 * (D[last][i] * D[last][i]);
                Cost += (-Y_output[i] * Math.Log(Yk[i]) - (1 - Y_output[i]) * Math.Log(1 - Yk[i]));
            }

            Cost /= input.y;
            L2 /= input.y;

            // Cleanup
            for (var layer = 0; layer < Weights.Count; layer++)
            {
                ManagedOps.Free(D[layer], X[layer], Z[layer]);
            }

            D.Clear();
            X.Clear();
            Z.Clear();

            ManagedOps.Free(Yk);
        }

        public void ApplyGradients(NeuralNetworkOptions opts)
        {
            for (var layer = 0; layer < Weights.Count; layer++)
            {
                ManagedMatrix.Add(Weights[layer], Deltas[layer], -opts.Alpha);

                // cleanup of arrays allocated in BackPropagation
                ManagedOps.Free(Deltas[layer]);
            }

            Deltas.Clear();
        }

        public void Rand(ManagedArray rand, Random random)
        {
            for (var x = 0; x < rand.Length(); x++)
            {
                rand[x] = (random.NextDouble() - 0.5) * 2.0;
            }
        }

        ManagedArray Labels(ManagedArray output, NeuralNetworkOptions opts)
        {
            var result = new ManagedArray(opts.Categories, opts.Items);
            var eye_matrix = ManagedMatrix.Diag(opts.Categories);

            for (var y = 0; y < opts.Items; y++)
            {
                if (opts.Categories > 1)
                {
                    for (var x = 0; x < opts.Categories; x++)
                    {
                        result[x, y] = eye_matrix[x, (int)output[y] - 1];
                    }
                }
                else
                {
                    result[y] = output[y];
                }
            }

            ManagedOps.Free(eye_matrix);

            return result;
        }

        public ManagedArray Predict(ManagedArray test, NeuralNetworkOptions opts)
        {
            Forward(test);

            var prediction = new ManagedArray(test.y);

            for (var y = 0; y < test.y; y++)
            {
                if (opts.Categories > 1)
                {
                    double maxval = Double.MinValue;

                    for (var x = 0; x < opts.Categories; x++)
                    {
                        double val = Yk[x, y];

                        if (val > maxval)
                        {
                            maxval = val;
                        }
                    }

                    prediction[y] = maxval;
                }
                else
                {
                    prediction[y] = Yk[y];
                }
            }

            // cleanup of arrays allocated in Forward
            ManagedOps.Free(Yk);

            // Cleanup
            for (var layer = 0; layer < Weights.Count; layer++)
            {
                ManagedOps.Free(X[layer], Z[layer]);
            }

            X.Clear();
            Z.Clear();

            return prediction;
        }

        public ManagedIntList Classify(ManagedArray test, NeuralNetworkOptions opts, double threshold = 0.5)
        {
            Forward(test);

            var classification = new ManagedIntList(test.y);

            for (var y = 0; y < test.y; y++)
            {
                if (opts.Categories > 1)
                {
                    var maxval = double.MinValue;
                    var maxind = 0;

                    for (var x = 0; x < opts.Categories; x++)
                    {
                        var val = Yk[x, y];

                        if (val > maxval)
                        {
                            maxval = val;
                            maxind = x;
                        }
                    }

                    classification[y] = maxind + 1;
                }
                else
                {
                    classification[y] = Yk[y] > threshold ? 1 : 0;
                }
            }

            // cleanup of arrays allocated in Forward
            ManagedOps.Free(Yk);

            for (var layer = 0; layer < Weights.Count; layer++)
            {
                ManagedOps.Free(X[layer], Z[layer]);
            }

            X.Clear();
            Z.Clear();

            return classification;
        }

        public void SetupLabels(ManagedArray output, NeuralNetworkOptions opts)
        {
            Y_output = Labels(output, opts);
        }

        public void Setup(ManagedArray output, NeuralNetworkOptions opts, bool Reset = true)
        {
            if (Reset)
            {
                if (Weights != null && Weights.Count > 0)
                {
                    for (var layer = 0; layer < Weights.Count; layer++)
                    {
                        ManagedOps.Free(Weights[layer]);
                    }

                    Weights.Clear();
                }

                Weights = new List<ManagedArray>
                {
                    new ManagedArray(opts.Inputs + 1, opts.Nodes)
                };

                for (var layer = 1; layer < opts.HiddenLayers; layer++)
                {
                    Weights.Add(new ManagedArray(opts.Nodes + 1, opts.Nodes));
                }

                Weights.Add(new ManagedArray(opts.Nodes + 1, opts.Categories));
            }

            SetupLabels(output, opts);

            var random = new Random(Guid.NewGuid().GetHashCode());

            if (Reset && Weights != null)
            {
                for (var layer = 0; layer < opts.HiddenLayers + 1; layer++)
                {
                    Rand(Weights[layer], random);
                }
            }

            Cost = 1.0;
            L2 = 1.0;

            Iterations = 0;
        }

        public bool Step(ManagedArray input, NeuralNetworkOptions opts)
        {
            Forward(input);
            BackPropagation(input);
            ApplyGradients(opts);

            Iterations = Iterations + 1;

            return (double.IsNaN(Cost) || Iterations >= opts.Epochs || Cost < opts.Tolerance);
        }

        public void Train(ManagedArray input, ManagedArray output, NeuralNetworkOptions opts)
        {
            Setup(output, opts);

            while (!Step(input, opts)) { }
        }

        public void Free()
        {
            ManagedOps.Free(Yk);
            ManagedOps.Free(Y_output);

            for (var layer = 0; layer < Weights.Count; layer++)
            {
                ManagedOps.Free(Weights[layer]);
            }

            Weights.Clear();
        }
    }
}
