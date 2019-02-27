using DeepLearnCS;
using Gdk;
using GLib;
using Gtk;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;

public partial class MainWindow : Gtk.Window
{
    Dialog Confirm;

    FileChooserDialog TextLoader, JsonLoader, JsonSaver, ImageSaver;
    string TrainingDataFileName, TestDataFileName, NetworkFileName, FileName;

    List<Delimiter> Delimiters = new List<Delimiter>();

    bool Paused = true;
    bool NetworkSetuped;
    bool NetworkLoaded;
    bool WeightsLoaded;

    Mutex Processing = new Mutex();

    int CurrentEpoch;
    bool TrainingDone;

    ManagedDNN Network = new ManagedDNN();
    NeuralNetworkOptions Options = new NeuralNetworkOptions();
    ManagedArray InputData = new ManagedArray();
    ManagedArray OutputData = new ManagedArray();
    ManagedArray TestData = new ManagedArray();
    ManagedArray NormalizationData = new ManagedArray();

    enum Pages
    {
        DATA = 0,
        TRAINING = 1,
        NETWORK = 2,
        PLOT = 3,
        ABOUT = 4
    };

    CultureInfo ci = new CultureInfo("en-us");

    public MainWindow() : base(Gtk.WindowType.Toplevel)
    {
        Build();

        InitializeUserInterface();
    }

    protected FileFilter AddFilter(string name, params string[] patterns)
    {
        var filter = new FileFilter { Name = name };

        foreach (var pattern in patterns)
            filter.AddPattern(pattern);

        return filter;
    }

    protected void InitializeUserInterface()
    {
        Title = "Deep Neural Network Classifier";

        Confirm = new Dialog(
            "Are you sure?",
            this,
            DialogFlags.Modal,
            "Yes", ResponseType.Accept,
            "No", ResponseType.Cancel
        )
        {
            Resizable = false,
            KeepAbove = true,
            TypeHint = WindowTypeHint.Dialog,
            WidthRequest = 250
        };

        Confirm.ActionArea.LayoutStyle = ButtonBoxStyle.Center;
        Confirm.WindowStateEvent += OnWindowStateEvent;

        TextLoader = new FileChooserDialog(
            "Load Text File",
            this,
            FileChooserAction.Open,
            "Cancel", ResponseType.Cancel,
            "Load", ResponseType.Accept
        );

        JsonLoader = new FileChooserDialog(
            "Load trained models",
            this,
            FileChooserAction.Open,
            "Cancel", ResponseType.Cancel,
            "Load", ResponseType.Accept
        );

        JsonSaver = new FileChooserDialog(
            "Save trained models",
            this,
            FileChooserAction.Save,
            "Cancel", ResponseType.Cancel,
            "Save", ResponseType.Accept
        );

        ImageSaver = new FileChooserDialog(
            "Save Filtered Image",
            this,
            FileChooserAction.Save,
            "Cancel", ResponseType.Cancel,
            "Save", ResponseType.Accept
        );

        TextLoader.AddFilter(AddFilter("Text files (csv/txt)", "*.txt", "*.csv"));
        TextLoader.Filter = TextLoader.Filters[0];

        JsonLoader.AddFilter(AddFilter("json", "*.json"));
        JsonSaver.AddFilter(AddFilter("json", "*.json"));

        Delimiters.Add(new Delimiter("Tab \\t", '\t'));
        Delimiters.Add(new Delimiter("Comma ,", ','));
        Delimiters.Add(new Delimiter("Space \\s", ' '));
        Delimiters.Add(new Delimiter("Vertical Pipe |", '|'));
        Delimiters.Add(new Delimiter("Colon :", ':'));
        Delimiters.Add(new Delimiter("Semi-Colon ;", ';'));
        Delimiters.Add(new Delimiter("Forward Slash /", '/'));
        Delimiters.Add(new Delimiter("Backward Slash \\", '\\'));

        ImageSaver.AddFilter(AddFilter("png", "*.png"));
        ImageSaver.AddFilter(AddFilter("jpg", "*.jpg", "*.jpeg"));
        ImageSaver.AddFilter(AddFilter("tif", "*.tif", "*.tiff"));
        ImageSaver.AddFilter(AddFilter("bmp", "*.bmp"));
        ImageSaver.AddFilter(AddFilter("ico", "*.ico"));
        ImageSaver.Filter = ImageSaver.Filters[0];

        UpdateDelimiterBox(DelimiterBox, Delimiters);

        UpdatePlotTypes();

        PlotImage.Pixbuf = Common.Pixbuf(PlotImage.WidthRequest, PlotImage.HeightRequest);

        Idle.Add(new IdleHandler(OnIdle));
    }

    protected void ToggleControls(bool toggle)
    {
        FilenameTrainingData.Sensitive = toggle;
        ViewTrainingData.Sensitive = toggle;
        OpenTrainingDataButton.Sensitive = toggle;
        ReloadTrainingDataButton.Sensitive = toggle;
        Examples.Sensitive = toggle;
        InputLayerNodes.Sensitive = toggle;
        Categories.Sensitive = toggle;

        FilenameTestData.Sensitive = toggle;
        ViewTestData.Sensitive = toggle;
        OpenTestDataButton.Sensitive = toggle;
        ReloadTestDataButton.Sensitive = toggle;
        Samples.Sensitive = toggle;

        DelimiterBox.Sensitive = toggle;

        LearningRate.Sensitive = toggle;
        HiddenLayerNodes.Sensitive = toggle;
        HiddenLayers.Sensitive = toggle;
        Tolerance.Sensitive = toggle;
        ViewClassification.Sensitive = toggle;
        Threshold.Sensitive = toggle;

        StartButton.Sensitive = toggle;
        StopButton.Sensitive = !toggle;
        ResetButton.Sensitive = toggle;
        ClassifyButton.Sensitive = toggle;

        HiddenLayerWeightSelector.Sensitive = toggle;
        ViewHiddenLayerWeights.Sensitive = toggle;
        ViewOutputLayerWeights.Sensitive = toggle;
        ViewNormalization.Sensitive = toggle;
        OpenNetworkButton.Sensitive = toggle;
        SaveNetworkButton.Sensitive = toggle;
        FilenameNetwork.Sensitive = toggle;

        LoadNetworkButton.Sensitive = toggle;

        PlotType.Sensitive = toggle;
        Feature1.Sensitive = toggle;
        Feature2.Sensitive = toggle;
        PlotButton.Sensitive = toggle;
        SavePlotButton.Sensitive = toggle;
    }

    protected void Pause()
    {
        if (Paused)
            return;

        Paused = true;

        ToggleControls(Paused);
    }

    protected void Run()
    {
        if (!Paused)
            return;

        Paused = false;

        ToggleControls(Paused);
    }

    protected string GetBaseFileName(string fullpath)
    {
        return System.IO.Path.GetFileNameWithoutExtension(fullpath);
    }

    protected string GetDirectory(string fullpath)
    {
        return System.IO.Path.GetDirectoryName(fullpath);
    }

    protected void ReloadTextFile(string FileName, TextView view, bool isTraining = false, SpinButton counter = null)
    {
        try
        {
            var current = DelimiterBox.Active;
            var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';

            var categories = new List<int>();

            if (File.Exists(FileName) && view != null)
            {
                var text = "";

                using (TextReader reader = File.OpenText(FileName))
                {
                    string line;
                    var count = 0;

                    while ((line = reader.ReadLine()) != null)
                    {
                        line = line.Trim();

                        if (!string.IsNullOrEmpty(line))
                        {
                            if (isTraining && counter != null)
                            {
                                var tokens = line.Split(delimiter);

                                if (tokens.Length > 1)
                                {
                                    var last = SafeConvert.ToInt32(tokens[tokens.Length - 1]);

                                    if (!categories.Contains(last) && last > 0)
                                    {
                                        categories.Add(last);
                                    }
                                }
                            }

                            text += count > 0 ? "\n" + line : line;

                            count++;
                        }
                    }
                }

                if (isTraining && counter != null)
                {
                    counter.Value = Convert.ToDouble(categories.Count, ci);
                }

                view.Buffer.Clear();

                view.Buffer.Text = text.Trim();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Error: {0}", ex.Message);
        }
    }

    protected void LoadTextFile(ref string FileName, string title, TextView view, Entry entry, bool isTraining = false, SpinButton counter = null)
    {
        TextLoader.Title = title;

        // Add most recent directory
        if (!string.IsNullOrEmpty(TextLoader.Filename))
        {
            var directory = System.IO.Path.GetDirectoryName(TextLoader.Filename);

            if (Directory.Exists(directory))
            {
                TextLoader.SetCurrentFolder(directory);
            }
        }

        if (TextLoader.Run() == (int)ResponseType.Accept)
        {
            if (!string.IsNullOrEmpty(TextLoader.Filename))
            {
                FileName = TextLoader.Filename;

                ReloadTextFile(FileName, view, isTraining, counter);

                if (entry != null)
                {
                    entry.Text = FileName;
                }
            }
        }

        TextLoader.Hide();
    }

    protected void LoadJson(ref string FileName, string title, Entry entry)
    {
        JsonLoader.Title = title;

        // Add most recent directory
        if (!string.IsNullOrEmpty(JsonLoader.Filename))
        {
            var directory = System.IO.Path.GetDirectoryName(JsonLoader.Filename);

            if (Directory.Exists(directory))
            {
                JsonLoader.SetCurrentFolder(directory);
            }
        }

        if (JsonLoader.Run() == (int)ResponseType.Accept)
        {
            if (!string.IsNullOrEmpty(JsonLoader.Filename))
            {
                FileName = JsonLoader.Filename;

                if (entry != null)
                {
                    entry.Text = FileName;
                }
            }
        }

        JsonLoader.Hide();
    }

    protected void SaveJson(ref string FileName, string title, Entry entry, string data)
    {
        JsonSaver.Title = title;

        JsonSaver.SelectFilename(FileName);

        string directory;

        // Add most recent directory
        if (!string.IsNullOrEmpty(JsonSaver.Filename))
        {
            directory = System.IO.Path.GetDirectoryName(JsonSaver.Filename);

            if (Directory.Exists(directory))
            {
                JsonSaver.SetCurrentFolder(directory);
            }
        }

        if (JsonSaver.Run() == (int)ResponseType.Accept)
        {
            if (!string.IsNullOrEmpty(JsonSaver.Filename))
            {
                FileName = JsonSaver.Filename;

                directory = GetDirectory(FileName);

                var ext = JsonSaver.Filter.Name;

                FileName = String.Format("{0}.{1}", GetBaseFileName(FileName), ext);

                if (!string.IsNullOrEmpty(data))
                {
                    var fullpath = String.Format("{0}/{1}", directory, FileName);

                    try
                    {
                        using (var file = new StreamWriter(fullpath, false))
                        {
                            file.Write(data);
                        }

                        FileName = fullpath;

                        entry.Text = FileName;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Error saving {0}: {1}", FileName, ex.Message);
                    }
                }
            }
        }

        JsonSaver.Hide();
    }

    protected string GetFileName(string fullpath)
    {
        return System.IO.Path.GetFileNameWithoutExtension(fullpath);
    }

    protected string GetName(string fullpath)
    {
        return System.IO.Path.GetFileName(fullpath);
    }

    protected void SavePlot()
    {
        ImageSaver.Title = "Save plot";

        string directory;

        // Add most recent directory
        if (!string.IsNullOrEmpty(ImageSaver.Filename))
        {
            directory = GetDirectory(ImageSaver.Filename);

            if (Directory.Exists(directory))
            {
                ImageSaver.SetCurrentFolder(directory);
            }
        }

        if (ImageSaver.Run() == (int)ResponseType.Accept)
        {
            if (!string.IsNullOrEmpty(ImageSaver.Filename))
            {
                FileName = ImageSaver.Filename;

                directory = GetDirectory(FileName);

                var ext = ImageSaver.Filter.Name;

                var fmt = ext;

                switch (ext)
                {
                    case "jpg":

                        if (!FileName.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) && !FileName.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase))
                        {
                            FileName = String.Format("{0}.jpg", GetFileName(FileName));
                        }

                        fmt = "jpeg";

                        break;

                    case "tif":

                        if (!FileName.EndsWith(".tif", StringComparison.OrdinalIgnoreCase) && !FileName.EndsWith(".tiff", StringComparison.OrdinalIgnoreCase))
                        {
                            FileName = String.Format("{0}.tif", GetFileName(FileName));
                        }

                        fmt = "tiff";

                        break;

                    default:

                        FileName = String.Format("{0}.{1}", GetFileName(FileName), ext);

                        break;
                }

                if (PlotImage.Pixbuf != null)
                {
                    FileName = GetName(FileName);

                    var fullpath = String.Format("{0}/{1}", directory, FileName);

                    try
                    {
                        PlotImage.Pixbuf.Save(fullpath, fmt);

                        FileName = fullpath;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Error saving {0}: {1}", FileName, ex.Message);
                    }
                }
            }
        }

        ImageSaver.Hide();
    }

    protected void UpdateDelimiterBox(ComboBox combo, List<Delimiter> delimeters)
    {
        combo.Clear();

        var cell = new CellRendererText();
        combo.PackStart(cell, false);
        combo.AddAttribute(cell, "text", 0);
        var store = new ListStore(typeof(string));
        combo.Model = store;

        foreach (var delimeter in delimeters)
        {
            store.AppendValues(delimeter.Name);
        }

        combo.Active = delimeters.Count > 0 ? 0 : -1;
    }

    protected void CopyToImage(Gtk.Image image, Pixbuf pixbuf, int OriginX, int OriginY)
    {
        if (pixbuf != null && image.Pixbuf != null)
        {
            image.Pixbuf.Fill(0);

            pixbuf.CopyArea(OriginX, OriginY, Math.Min(image.WidthRequest, pixbuf.Width), Math.Min(image.HeightRequest, pixbuf.Height), image.Pixbuf, 0, 0);

            image.QueueDraw();
        }
    }

    protected void PlotNetwork(int f1 = 0, int f2 = 1)
    {
        var test = ViewTestData.Buffer.Text.Trim();

        if (string.IsNullOrEmpty(test))
            return;

        var type = PlotType.Active;

        if (NetworkSetuped && SetupTestData(test) && type >= 0 && type < 2)
        {
            var TestOptions = Options;

            TestOptions.Items = TestData.y;

            Pixbuf pixbuf;

            switch (type)
            {
                case 0:

                    pixbuf = Plot.Points(Network, TestOptions, Threshold.Value / 100, TestData, PlotImage.WidthRequest, PlotImage.HeightRequest, f1, f2);

                    break;

                case 1:

                    pixbuf = Plot.Contour(Network, TestOptions, Threshold.Value / 100, TestData, PlotImage.WidthRequest, PlotImage.HeightRequest, f1, f2);

                    break;

                default:

                    pixbuf = Plot.Points(Network, TestOptions, Threshold.Value / 100, TestData, PlotImage.WidthRequest, PlotImage.HeightRequest, f1, f2);

                    break;
            }

            if (pixbuf != null)
            {
                CopyToImage(PlotImage, pixbuf, 0, 0);

                Common.Free(pixbuf);
            }
        }
    }

    protected void UpdatePlotTypes()
    {
        PlotType.Clear();

        var cell = new CellRendererText();
        PlotType.PackStart(cell, false);
        PlotType.AddAttribute(cell, "text", 0);
        var store = new ListStore(typeof(string));

        PlotType.Model = store;

        store.AppendValues("Data points");
        store.AppendValues("Class boundaries");

        PlotType.Active = 0;
    }

    protected void ReparentTextView(Fixed parent, ScrolledWindow window, int x, int y)
    {
        var source = (Fixed)window.Parent;
        source.Remove(window);

        parent.Add(window);

        Fixed.FixedChild child = ((Fixed.FixedChild)(parent[window]));

        child.X = x;
        child.Y = y;
    }

    protected void ReparentLabel(Fixed parent, Label label, int x, int y)
    {
        label.Reparent(parent);

        parent.Move(label, x, y);
    }

    protected void UpdateParameters(TextView text, SpinButton counter, SpinButton counter2, bool isTraining = true)
    {
        var input = text.Buffer;

        var current = DelimiterBox.Active;

        var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';

        if (input.LineCount > 0)
        {
            counter.Value = input.LineCount;

            bool first = false;

            using (StringReader reader = new StringReader(input.Text.Trim()))
            {
                var line = reader.ReadLine();

                if (!string.IsNullOrEmpty(line))
                {
                    if (!first)
                        first = true;

                    var tokens = line.Split(delimiter);

                    if (first)
                    {
                        if (isTraining && counter2 != null && tokens.Length > 0)
                        {
                            counter2.Value = tokens.Length - 1;
                        }
                    }
                }
            }
        }
    }

    protected void UpdateHiddenLayerWeightsSelector(ComboBox combo, ManagedArray[] Weights)
    {
        if (Weights != null)
        {
            combo.Clear();

            var cell = new CellRendererText();
            combo.PackStart(cell, false);
            combo.AddAttribute(cell, "text", 0);
            var store = new ListStore(typeof(string));
            combo.Model = store;

            for (var layer = 0; layer < Weights.GetLength(0) - 1; layer++)
            {
                store.AppendValues(string.Format("Hidden layer {0}", layer + 1));
            }

            combo.Active = Weights.GetLength(0) > 0 ? 0 : -1;
        }
    }

    protected void UpdateHiddenLayerWeights(ManagedArray[] Weights)
    {
        if (Weights != null & Weights.GetLength(0) > 1)
        {
            var current = HiddenLayerWeightSelector.Active;

            if (current >= 0 && current < Weights.GetLength(0))
            {
                UpdateTextView(ViewHiddenLayerWeights, Weights[current]);
            }
        }
    }

    protected void UpdateOutputLayerWeights(ManagedArray[] Weights)
    {
        if (Weights != null & Weights.GetLength(0) > 1)
        {
            UpdateTextView(ViewOutputLayerWeights, Weights[Weights.GetLength(0) - 1]);
        }
    }

    protected void UpdateNetworkWeights()
    {
        if (NetworkSetuped)
        {
            UpdateHiddenLayerWeightsSelector(HiddenLayerWeightSelector, Network.Weights);
            UpdateHiddenLayerWeights(Network.Weights);
            UpdateOutputLayerWeights(Network.Weights);
        }
    }

    protected void UpdateClassifierInfo()
    {
        if (NetworkSetuped)
        {
            Iterations.Text = Network.Iterations.ToString(ci);

            ErrorCost.Text = Network.Cost.ToString("0.#####e+00", ci);
            L2.Text = Network.L2.ToString("0.#####e+00", ci);
        }
    }

    protected void UpdateProgressBar()
    {
        if (Epochs.Value > 0)
        {
            ProgressBar.Fraction = Math.Round(CurrentEpoch / Epochs.Value, 2);

            ProgressBar.Text = TrainingDone ? "Done" : string.Format("Training ({0}%)...", Convert.ToInt32(ProgressBar.Fraction * 100, ci));
        }
    }

    protected void UpdateTrainingDisplay()
    {
        UpdateClassifierInfo();
        UpdateProgressBar();
    }

    protected void Classify()
    {
        var test = ViewTestData.Buffer.Text.Trim();

        if (string.IsNullOrEmpty(test))
            return;

        if (NetworkSetuped && SetupTestData(test))
        {
            var TestOptions = Options;

            TestOptions.Items = TestData.y;

            var classification = Network.Classify(TestData, TestOptions, Threshold.Value / 100);

            ViewClassification.Buffer.Clear();

            string text = "";

            for (var i = 0; i < classification.x; i++)
            {
                text += Convert.ToString(classification[i], ci);

                if (i < classification.x - 1)
                    text += "\n";
            }

            ViewClassification.Buffer.Text = text;

            classification.Free();
        }

        ViewTestData.Buffer.Text = test;
    }

    protected void NormalizeData(ManagedArray input, ManagedArray normalization)
    {
        for (int y = 0; y < input.y; y++)
        {
            for (int x = 0; x < input.x; x++)
            {
                var min = normalization[x, 0];
                var max = normalization[x, 1];

                input[x, y] = (input[x, y] - min) / (max - min);
            }
        }
    }

    protected bool SetupInputData(string training)
    {
        var text = training.Trim();

        if (string.IsNullOrEmpty(text))
            return false;

        var TrainingBuffer = new TextBuffer(new TextTagTable())
        {
            Text = text
        };

        Examples.Value = Convert.ToDouble(TrainingBuffer.LineCount, ci);

        var inpx = Convert.ToInt32(InputLayerNodes.Value, ci);
        var inpy = Convert.ToInt32(Examples.Value, ci);

        ManagedOps.Free(InputData, OutputData, NormalizationData);

        InputData = new ManagedArray(inpx, inpy);
        NormalizationData = new ManagedArray(inpx, 2);
        OutputData = new ManagedArray(1, inpy);

        int min = 0;
        int max = 1;

        for (int x = 0; x < inpx; x++)
        {
            NormalizationData[x, min] = double.MaxValue;
            NormalizationData[x, max] = double.MinValue;
        }

        var current = DelimiterBox.Active;
        var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';
        var inputs = inpx;

        using (var reader = new StringReader(TrainingBuffer.Text))
        {
            for (int y = 0; y < inpy; y++)
            {
                var line = reader.ReadLine();

                if (!string.IsNullOrEmpty(line))
                {
                    var tokens = line.Split(delimiter);

                    if (inputs > 0 && tokens.Length > inputs)
                    {
                        OutputData[0, y] = SafeConvert.ToDouble(tokens[inputs]);

                        for (int x = 0; x < inpx; x++)
                        {
                            var data = SafeConvert.ToDouble(tokens[x]);

                            NormalizationData[x, min] = data < NormalizationData[x, min] ? data : NormalizationData[x, min];
                            NormalizationData[x, max] = data > NormalizationData[x, max] ? data : NormalizationData[x, max];

                            InputData[x, y] = data;
                        }
                    }
                }
            }
        }

        NormalizeData(InputData, NormalizationData);

        return true;
    }

    protected bool SetupTestData(string test)
    {
        var text = test.Trim();

        if (string.IsNullOrEmpty(text))
            return false;

        var TestBuffer = new TextBuffer(new TextTagTable())
        {
            Text = text
        };

        Samples.Value = Convert.ToDouble(TestBuffer.LineCount, ci);

        var inpx = Convert.ToInt32(InputLayerNodes.Value, ci);
        var tsty = Convert.ToInt32(Samples.Value, ci);

        ManagedOps.Free(TestData);

        TestData = new ManagedArray(inpx, tsty);

        var current = DelimiterBox.Active;
        var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';
        var inputs = inpx;

        using (var reader = new StringReader(TestBuffer.Text))
        {
            for (int y = 0; y < tsty; y++)
            {
                var line = reader.ReadLine();

                if (!string.IsNullOrEmpty(line))
                {
                    var tokens = line.Split(delimiter);

                    if (inputs > 0 && tokens.Length >= inpx)
                    {
                        for (int x = 0; x < inpx; x++)
                        {
                            TestData[x, y] = SafeConvert.ToDouble(tokens[x]);
                        }
                    }
                }
            }
        }

        NormalizeData(TestData, NormalizationData);

        UpdateTextView(ViewNormalization, NormalizationData);

        return true;
    }

    protected void SetupNetworkTraining()
    {
        NetworkSetuped = false;

        var training = ViewTrainingData.Buffer.Text.Trim();

        if (string.IsNullOrEmpty(training))
            return;

        NetworkSetuped = SetupInputData(training);

        // Reset Network
        Network.Free();

        Options.Alpha = Convert.ToDouble(LearningRate.Value, ci) / 100;
        Options.Epochs = Convert.ToInt32(Epochs.Value, ci);
        Options.Inputs = Convert.ToInt32(InputLayerNodes.Value, ci);
        Options.Categories = Convert.ToInt32(Categories.Value, ci);
        Options.Items = InputData.y;
        Options.Nodes = Convert.ToInt32(HiddenLayerNodes.Value, ci);
        Options.HiddenLayers = Convert.ToInt32(HiddenLayers.Value, ci);
        Options.Tolerance = Convert.ToDouble(Tolerance.Value, ci) / 100000;
        Options.UseL2 = UseL2.Active;

        if (UseOptimizer.Active)
        {
            Network.SetupOptimizer(InputData, OutputData, Options, true);
        }
        else
        {
            Network.Setup(OutputData, Options, true);
        }

        ViewTrainingData.Buffer.Text = training;
    }

    protected void SetupNetwork()
    {
        NetworkSetuped = false;

        if (WeightsLoaded)
        {
            var training = ViewTrainingData.Buffer.Text.Trim();

            if (!string.IsNullOrEmpty(training))
            {
                SetupInputData(training);
            }

            Options.Alpha = Convert.ToDouble(LearningRate.Value, ci) / 100;
            Options.Epochs = Convert.ToInt32(Epochs.Value, ci);
            Options.Inputs = Convert.ToInt32(InputLayerNodes.Value, ci);
            Options.Categories = Convert.ToInt32(Categories.Value, ci);
            Options.Items = InputData.y;
            Options.Nodes = Convert.ToInt32(HiddenLayerNodes.Value, ci);
            Options.HiddenLayers = Convert.ToInt32(HiddenLayers.Value, ci);
            Options.Tolerance = Convert.ToDouble(Tolerance.Value, ci) / 100000;
            Options.UseL2 = UseL2.Active;

            ProgressBar.Fraction = 1.0;
            ProgressBar.Text = "Network Loaded";

            NetworkLoaded = WeightsLoaded;

            if (UseOptimizer.Active)
            {
                Network.SetupOptimizer(InputData, OutputData, Options, false);
            }
            else
            {
                Network.Setup(OutputData, Options, false);
            }

            NetworkSetuped = true;

            CurrentEpoch = 0;

            Iterations.Text = "";
            ErrorCost.Text = "";
            L2.Text = "";

            UseOptimizer.Sensitive = true;
            UseL2.Sensitive = true;
            Epochs.Sensitive = true;
            WeightsLoaded = false;
        }
    }

    protected void UpdateTextView(TextView view, ManagedArray data)
    {
        if (data != null)
        {
            var current = DelimiterBox.Active;
            var delimiter = current >= 0 && current < Delimiters.Count ? Delimiters[current].Character : '\t';

            view.Buffer.Clear();

            var text = "";

            for (int y = 0; y < data.y; y++)
            {
                if (y > 0)
                    text += "\n";

                for (int x = 0; x < data.x; x++)
                {
                    if (x > 0)
                        text += delimiter;

                    text += data[x, y].ToString(ci);
                }
            }

            view.Buffer.Text = text;
        }
    }

    protected bool GetConfirmation()
    {
        var confirm = Confirm.Run() == (int)ResponseType.Accept;

        Confirm.Hide();

        return confirm;
    }

    protected void CleanShutdown()
    {
        // Clean-Up Routines Here
        Network.Free();

        ManagedOps.Free(InputData, OutputData, TestData, NormalizationData);

        Common.Free(PlotImage.Pixbuf);
        Common.Free(PlotImage);

        Plot.Free();
    }

    protected void Quit()
    {
        CleanShutdown();

        Application.Quit();
    }

    protected void OnWindowStateEvent(object sender, WindowStateEventArgs args)
    {
        var state = args.Event.NewWindowState;

        if (state == WindowState.Iconified)
        {
            Confirm.Hide();
        }

        args.RetVal = true;
    }

    protected void OnAboutButtonClicked(object sender, EventArgs e)
    {
        MainNotebook.Page = (int)Pages.ABOUT;
    }

    protected void OnQuitButtonClicked(object sender, EventArgs e)
    {
        OnDeleteEvent(sender, new DeleteEventArgs());
    }

    protected void OnDeleteEvent(object sender, DeleteEventArgs a)
    {
        if (GetConfirmation())
        {
            Quit();
        }

        a.RetVal = true;
    }

    bool OnIdle()
    {
        Processing.WaitOne();

        if (!Paused && NetworkSetuped)
        {
            var result = UseOptimizer.Active ? Network.StepOptimizer(InputData, Options) : Network.Step(InputData, Options);

            CurrentEpoch = Network.Iterations;

            if (result)
            {
                var Epoch = Convert.ToInt32(Epochs.Value, ci);

                UpdateNetworkWeights();

                CurrentEpoch = Epoch;

                TrainingDone = true;

                NetworkLoaded = false;

                Classify();

                UseOptimizer.Sensitive = true;
                UseL2.Sensitive = true;
                Epochs.Sensitive = true;

                UpdateTrainingDisplay();

                Pause();
            }

            if (CurrentEpoch % 1000 == 0)
            {
                UpdateTrainingDisplay();
            }
        }

        Processing.ReleaseMutex();

        return true;
    }

    protected void OnOpenTrainingDataButtonClicked(object sender, EventArgs e)
    {
        LoadTextFile(ref TrainingDataFileName, "Load Training Data", ViewTrainingData, FilenameTrainingData, true, Categories);

        UpdateParameters(ViewTrainingData, Examples, InputLayerNodes, true);

        HiddenLayerNodes.Value = 2 * InputLayerNodes.Value;
    }

    protected void OnReloadTrainingDataButtonClicked(object sender, EventArgs e)
    {
        if (!string.IsNullOrEmpty(TrainingDataFileName))
            ReloadTextFile(TrainingDataFileName, ViewTrainingData, true, Categories);

        UpdateParameters(ViewTrainingData, Examples, InputLayerNodes, true);

        HiddenLayerNodes.Value = 2 * InputLayerNodes.Value;
    }

    protected void OnOpenTestDataButtonClicked(object sender, EventArgs e)
    {
        LoadTextFile(ref TestDataFileName, "Load Test Data", ViewTestData, FilenameTestData, false, null);

        UpdateParameters(ViewTestData, Samples, null, false);
    }

    protected void OnReloadTestDataButtonClicked(object sender, EventArgs e)
    {
        if (!string.IsNullOrEmpty(TestDataFileName))
            ReloadTextFile(TestDataFileName, ViewTestData);

        UpdateParameters(ViewTestData, Samples, null, false);
    }

    protected void OnMainNotebookSwitchPage(object o, SwitchPageArgs args)
    {
        switch (args.PageNum)
        {
            case (int)Pages.DATA:

                ReparentTextView(LayoutPageData, WindowTestData, 30, 270);
                ReparentLabel(LayoutPageData, LabelTestData, 30, 220);

                break;

            case (int)Pages.TRAINING:

                ReparentTextView(LayoutPageTraining, WindowTestData, 30, 310);
                ReparentLabel(LayoutPageTraining, LabelTestData, 30, 290);

                break;

            default:

                ReparentTextView(LayoutPageData, WindowTestData, 30, 270);
                ReparentLabel(LayoutPageData, LabelTestData, 30, 220);

                break;
        }
    }

    protected void OnStartButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        if (TrainingDone)
        {
            TrainingDone = false && !NetworkLoaded;

            NetworkSetuped = false || NetworkLoaded;

            CurrentEpoch = 0;

            Epochs.Sensitive = !NetworkLoaded;
        }

        if (!NetworkSetuped)
        {
            Epochs.Sensitive = false;
            UseOptimizer.Sensitive = false;
            UseL2.Sensitive = false;

            SetupNetworkTraining();

            ViewClassification.Buffer.Clear();

            CurrentEpoch = Network.Iterations;
        }

        UpdateProgressBar();

        if (NetworkSetuped)
            Run();
    }

    protected void OnStopButtonClicked(object sender, EventArgs e)
    {
        if (Paused)
            return;

        UpdateTrainingDisplay();

        Pause();
    }

    protected void OnResetButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        CurrentEpoch = 0;

        UpdateProgressBar();

        Iterations.Text = "";
        ErrorCost.Text = "";
        L2.Text = "";

        NetworkSetuped = false;
        NetworkLoaded = false;
        WeightsLoaded = false;

        TrainingDone = false;

        UseOptimizer.Sensitive = true;
        UseL2.Sensitive = true;

        Epochs.Sensitive = true;

        ProgressBar.Text = "";
    }

    protected void OnClassifyButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        Classify();
    }

    protected void OnLoadNetworkButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        SetupNetwork();
    }

    protected void OnOpenNetworkButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        LoadJson(ref NetworkFileName, "Load network model", FilenameNetwork);

        Network.Free();

        Network = Utility.LoadDNN(GetDirectory(NetworkFileName), GetBaseFileName(NetworkFileName), NormalizationData);

        if (Network != null)
        {
            if (Network.Weights != null && Network.Weights.GetLength(0) > 1)
            {
                if (Network.Weights[0].x > 0)
                    InputLayerNodes.Value = Network.Weights[0].x - 1;

                if (Network.Weights[0].y > 0)
                    HiddenLayerNodes.Value = Network.Weights[0].y;

                HiddenLayers.Value = Network.Weights.GetLength(0) - 1;

                if (Network.Weights[Network.Weights.GetLength(0) - 1].y > 0)
                    Categories.Value = Network.Weights[Network.Weights.GetLength(0) - 1].y;

                UpdateHiddenLayerWeightsSelector(HiddenLayerWeightSelector, Network.Weights);
                UpdateHiddenLayerWeights(Network.Weights);
                UpdateOutputLayerWeights(Network.Weights);

                if (NormalizationData != null)
                    UpdateTextView(ViewNormalization, NormalizationData);

                WeightsLoaded = true;
                TrainingDone = true;
            }
        }
        else
        {
            Network = new ManagedDNN();
        }
    }

    protected void OnSaveNetworkButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        if (NetworkSetuped)
        {
            var json = Utility.Serialize(Network, NormalizationData);

            SaveJson(ref NetworkFileName, "Save network model", FilenameNetwork, json);
        }
    }

    protected void OnUseL2Toggled(object sender, EventArgs e)
    {
        Options.UseL2 = UseL2.Active;
        
        if (UseL2.Active)
        {
            UseOptimizer.Active = false;
        }
    }

    protected void OnUseOptimizerToggled(object sender, EventArgs e)
    {
        if (UseOptimizer.Active)
        {
            UseL2.Active = false;
        }
    }

    protected void OnHiddenLayerWeightSelectorChanged(object sender, EventArgs e)
    {
        if (!Paused || (!NetworkSetuped && !WeightsLoaded))
            return;

        UpdateHiddenLayerWeights(Network.Weights);
    }

    protected void OnPlotButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        var feature1 = Convert.ToInt32(Feature1.Value);
        var feature2 = Convert.ToInt32(Feature2.Value);
        var features = Convert.ToInt32(InputLayerNodes.Value);

        if (feature1 >= 0 && feature1 < features && feature2 >= 0 && feature2 < features && feature1 != feature2)
        {
            ToggleControls(false);

            PlotNetwork(feature1, feature2);

            ToggleControls(true);
        }
    }

    protected void OnSavePlotButtonClicked(object sender, EventArgs e)
    {
        if (!Paused)
            return;

        SavePlot();
    }
}
