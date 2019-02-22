using System;
using System.Collections.Generic;

namespace DeepLearnCS
{
    public class ManagedDNN
    {
        public ManagedArray[] Weights;
        public ManagedArray[] Deltas;

        public ManagedArray Y;
        public ManagedArray Y_true;

        // intermediate results
        public ManagedArray[] X;
        public ManagedArray[] Z;

        // internal use
        private ManagedArray[] Activations;
        private ManagedArray[] D;

        // Error
        public double Cost;
        public double L2;

        public bool UseL2;

        public int Iterations;

        Optimize Optimizer = new Optimize();

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
                L2 += 0.5 * (D[0][i] * D[0][i]);
                Cost += (-Y_true[i] * Math.Log(Y[i]) - (1 - Y_true[i]) * Math.Log(1 - Y[i]));
            }

            Cost /= input.y;
            L2 /= input.y;

            // Cleanup
            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                ManagedOps.Free(D[layer], X[layer], Z[layer]);
            }
        }

        public void ClearDeltas()
        {
            for (var layer = 0; layer < Weights.GetLength(0); layer++)
            {
                // cleanup of arrays allocated in BackPropagation
                ManagedOps.Free(Deltas[layer]);
            }
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

                if (D != null && D.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < D.GetLength(0); layer++)
                    {
                        ManagedOps.Free(D[layer]);
                    }
                }

                if (Deltas != null && Deltas.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < Deltas.GetLength(0); layer++)
                    {
                        ManagedOps.Free(Deltas[layer]);
                    }
                }

                if (X != null && X.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < X.GetLength(0); layer++)
                    {
                        ManagedOps.Free(X[layer]);
                    }
                }

                if (Z != null && Z.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < Z.GetLength(0); layer++)
                    {
                        ManagedOps.Free(Z[layer]);
                    }
                }

                if (Weights != null && Weights.GetLength(0) > 0)
                {
                    for (var layer = 0; layer < Weights.GetLength(0); layer++)
                    {
                        ManagedOps.Free(Weights[layer]);
                    }
                }

                Weights = new ManagedArray[opts.HiddenLayers + 1];

                Weights[0] = new ManagedArray(opts.Inputs + 1, opts.Nodes);

                for (var layer = 1; layer < opts.HiddenLayers; layer++)
                {
                    Weights[layer] = (new ManagedArray(opts.Nodes + 1, opts.Nodes));
                }

                Weights[opts.HiddenLayers] = (new ManagedArray(opts.Nodes + 1, opts.Categories));
            }

            Activations = new ManagedArray[opts.HiddenLayers];
            Deltas = new ManagedArray[opts.HiddenLayers + 1];
            X = new ManagedArray[opts.HiddenLayers + 1];
            D = new ManagedArray[opts.HiddenLayers + 1];
            Z = new ManagedArray[opts.HiddenLayers + 1];

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

            var optimized = (double.IsNaN(UseL2 ? L2 : Cost) || (UseL2 ? L2 : Cost) < opts.Tolerance);

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

        // Reshape Network Weights for use in optimizer
        public double[] ReshapeWeights(ManagedArray[] A)
        {
            var size = 0;

            if (A != null && A.GetLength(0) > 0)
            {
                for (var layer = 0; layer < A.GetLength(0); layer++)
                {
                    size += A[layer].x * A[layer].y;
                }
            }

            var XX = new double[size];

            if (A != null && A.GetLength(0) > 0)
            {
                var index = 0;

                for (var layer = 0; layer < A.GetLength(0); layer++)
                {
                    for (var x = 0; x < A[layer].x; x++)
                    {
                        for (var y = 0; y < A[layer].y; y++)
                        {
                            XX[index] = A[layer][x, y];

                            index++;
                        }
                    }
                }
            }

            return XX;
        }

        // Transform vector back into Network Weights
        public void ReshapeWeights(double[] XX, ManagedArray[] A)
        {
            var index = 0;

            for (var layer = 0; layer < A.GetLength(0); layer++)
            {
                for (var x = 0; x < A[layer].x; x++)
                {
                    for (var y = 0; y < A[layer].y; y++)
                    {
                        if (index < XX.Length)
                            A[layer][x, y] = XX[index];

                        index++;
                    }
                }
            }
        }

        ManagedArray OptimizerInput;

        public FuncOutput OptimizerCost(double[] XX)
        {
            ReshapeWeights(XX, Weights);

            if (OptimizerInput != null)
                Forward(OptimizerInput);

            if (OptimizerInput != null)
                BackPropagation(OptimizerInput);

            XX = ReshapeWeights(Deltas);

            ClearDeltas();

            return new FuncOutput(Cost, XX);
        }

        public void SetupOptimizer(ManagedArray input, ManagedArray output, NeuralNetworkOptions opts, bool Reset = true)
        {
            Setup(output, opts, Reset);

            Optimizer.MaxIterations = opts.Epochs;

            var XX = ReshapeWeights(Weights);

            OptimizerInput = input;

            Optimizer.Setup(OptimizerCost, XX);
        }

        public bool StepOptimizer(ManagedArray input, NeuralNetworkOptions opts)
        {
            OptimizerInput = input;

            var XX = ReshapeWeights(Weights);

            Optimizer.Step(OptimizerCost, XX);

            Iterations = Optimizer.Iterations;

            Cost = Optimizer.f1;

            OptimizerInput = null;

            return (double.IsNaN(Cost) || Iterations >= opts.Epochs || (Cost) < opts.Tolerance);
        }

        public void Optimize(ManagedArray input, ManagedArray output, NeuralNetworkOptions opts)
        {
            SetupOptimizer(input, output, opts);

            while (!StepOptimizer(input, opts)) { }
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
        }
    }
}
