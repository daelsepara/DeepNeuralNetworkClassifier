using System;
using System.Collections.Generic;

namespace DeepLearnCS
{
    public class ManagedDNN
    {
        //public List<ManagedArray> Weights = new List<ManagedArray>();
        //public List<ManagedArray> Deltas = new List<ManagedArray>();
        public ManagedArray[] Weights;
        public ManagedArray[] Deltas;

        public ManagedArray Y;
        public ManagedArray Y_true;

        // intermediate results
        //public List<ManagedArray> X = new List<ManagedArray>();
        //public List<ManagedArray> Z = new List<ManagedArray>();
        public ManagedArray[] X;
        public ManagedArray[] Z;

        // internal use
        //private List<ManagedArray> Activations = new List<ManagedArray>();
        //private List<ManagedArray> D = new List<ManagedArray>();
        private ManagedArray[] Activations;
        private ManagedArray[] D;

        // Error
        public double Cost;
        public double L2;

        public int Iterations;

        // Forward Propagation
        public void Forward(ManagedArray input)
        {
            // create bias column
            var InputBias = new ManagedArray(1, input.y, false);
            ManagedOps.Set(InputBias, 1.0);

            // Compute input activations
            var last = Weights.GetLength(0) - 1;

            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                var XX = layer == 0 ? ManagedMatrix.CBind(InputBias, input) : ManagedMatrix.CBind(InputBias, Activations[layer - 1]);
                var tW = ManagedMatrix.Transpose(Weights[layer]);
                var ZZ = ManagedMatrix.Multiply(XX, tW);

                X[layer] = XX;
                Z[layer] = ZZ;

                if (layer != last)
                {
                    var SS = ManagedMatrix.Sigm(ZZ);

                    Activations[layer] = SS;
                }
                else
                {
                    ManagedOps.Free(Y);

                    Y = ManagedMatrix.Sigm(ZZ);
                }

                ManagedOps.Free(tW);
            }

            // Cleanup
            for (var layer = 0; layer < Activations.GetLength(0); layer++)
            {
                ManagedOps.Free(Activations[layer]);
            }

            //Activations.Clear();

            ManagedOps.Free(InputBias);
        }

        // Backward propagation
        public void BackPropagation(ManagedArray input)
        {
            var last = Weights.GetLength(0) - 1;

            D[0] = ManagedMatrix.Diff(Y, Y_true);

            var current = 1;

            for (var layer = last - 1; layer >= 0; layer--)
            {
                var prev = current - 1;

                var W = new ManagedArray(Weights[layer + 1].x - 1, Weights[layer + 1].y, false);
                var DZ = ManagedMatrix.DSigm(Z[layer]);

                D[current] = (new ManagedArray(W.x, D[prev].y, false));

                ManagedOps.Copy2D(W, Weights[layer + 1], 1, 0);
                ManagedMatrix.Multiply(D[current], D[prev], W);
                ManagedMatrix.Product(D[current], DZ);

                ManagedOps.Free(W, DZ);

                current++;
            }

            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                var tD = ManagedMatrix.Transpose(D[Weights.GetLength(0) - layer - 1]);

                Deltas[layer] = (new ManagedArray(Weights[layer].x, Weights[layer].y, false));

                ManagedMatrix.Multiply(Deltas[layer], tD, X[layer]);
                ManagedMatrix.Multiply(Deltas[layer], 1.0 / input.y);

                ManagedOps.Free(tD);
            }

            Cost = 0.0;
            L2 = 0.0;

            for (var i = 0; i < Y_true.Length(); i++)
            {
                L2 += 0.5 * (D[last][i] * D[last][i]);
                Cost += (-Y_true[i] * Math.Log(Y[i]) - (1 - Y_true[i]) * Math.Log(1 - Y[i]));
            }

            Cost /= input.y;
            L2 /= input.y;

            // Cleanup
            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                ManagedOps.Free(D[layer], X[layer], Z[layer]);
            }

            //D.Clear();
            //X.Clear();
            //Z.Clear();
        }

        public void ClearDeltas()
        {
            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                // cleanup of arrays allocated in BackPropagation
                ManagedOps.Free(Deltas[layer]);
            }

            //Deltas.Clear();
        }

        public void ApplyGradients(NeuralNetworkOptions opts)
        {
            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                ManagedMatrix.Add(Weights[layer], Deltas[layer], -opts.Alpha);
            }
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
            var result = new ManagedArray(opts.Categories, opts.Items, false);
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
                        double val = Y[x, y];

                        if (val > maxval)
                        {
                            maxval = val;
                        }
                    }

                    prediction[y] = maxval;
                }
                else
                {
                    prediction[y] = Y[y];
                }
            }

            // cleanup of arrays allocated in Forward propagation
            ManagedOps.Free(Y);

            // Cleanup
            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                ManagedOps.Free(X[layer], Z[layer]);
            }

            //X.Clear();
            //Z.Clear();

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
                        var val = Y[x, y];

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
                    classification[y] = Y[y] > threshold ? 1 : 0;
                }
            }

            // cleanup of arrays allocated in Forward propagation
            ManagedOps.Free(Y);

            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                ManagedOps.Free(X[layer], Z[layer]);
            }

            //X.Clear();
            //Z.Clear();

            return classification;
        }

        public void SetupLabels(ManagedArray output, NeuralNetworkOptions opts)
        {
            Y_true = Labels(output, opts);
        }

        public void Setup(ManagedArray output, NeuralNetworkOptions opts, bool Reset = true)
        {
            if (Reset)
            {
                if (Activations != null && Activations.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < Activations.GetLength(0); layer++)
                    {
                        ManagedOps.Free(Activations[layer]);
                    }
                }

                Activations = new ManagedArray[opts.HiddenLayers];

                if (D != null && D.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < D.GetLength(0); layer++)
                    {
                        ManagedOps.Free(D[layer]);
                    }
                }

                D = new ManagedArray[opts.HiddenLayers + 1];

                if (Deltas != null && Deltas.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < Deltas.GetLength(0); layer++)
                    {
                        ManagedOps.Free(Deltas[layer]);
                    }
                }

                Deltas = new ManagedArray[opts.HiddenLayers + 1];

                if (X != null && X.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < X.GetLength(0); layer++)
                    {
                        ManagedOps.Free(X[layer]);
                    }
                }

                X = new ManagedArray[opts.HiddenLayers + 1];

                if (Z != null && Z.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < Z.GetLength(0); layer++)
                    {
                        ManagedOps.Free(Z[layer]);
                    }
                }

                Z = new ManagedArray[opts.HiddenLayers + 1];

                if (Weights != null && Weights.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < Weights.GetLength(0); layer++)
                    {
                        ManagedOps.Free(Weights[layer]);
                    }

                    //Weights.Clear();
                }

                Weights = new ManagedArray[opts.HiddenLayers + 1];
                Weights[0] = new ManagedArray(opts.Inputs + 1, opts.Nodes);

                /*
                Weights = new List<ManagedArray>
                {
                    new ManagedArray(opts.Inputs + 1, opts.Nodes)
                };
                */

                for (var layer = 1; layer < opts.HiddenLayers; layer++)
                {
                    Weights[layer] = (new ManagedArray(opts.Nodes + 1, opts.Nodes));
                }

                Weights[opts.HiddenLayers] = (new ManagedArray(opts.Nodes + 1, opts.Categories));
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

            var optimized = (double.IsNaN(Cost) || Cost < opts.Tolerance);

            // Apply gradients only if the error is still high
            if (!optimized)
            {
                ApplyGradients(opts);
            }

            ClearDeltas();

            Iterations = Iterations + 1;

            return (optimized || Iterations >= opts.Epochs);
        }

        public void Train(ManagedArray input, ManagedArray output, NeuralNetworkOptions opts)
        {
            Setup(output, opts);

            while (!Step(input, opts)) { }
        }

        public void Free()
        {
            ManagedOps.Free(Y);
            ManagedOps.Free(Y_true);

            if (Weights != null)
            {
                for (var layer = 0; layer < Weights.GetLength(0); layer++)
                {
                    ManagedOps.Free(Weights[layer]);
                }
            }

            //Weights.Clear();
        }
    }
}
