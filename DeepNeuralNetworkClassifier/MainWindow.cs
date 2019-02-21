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
    string TrainingSetFileName, TestSetFileName;
    string WeightsFileName;

    List<Delimiter> Delimiters = new List<Delimiter>();

    bool Paused = true;
    bool NetworkSetuped;
    bool NetworkLoaded;

    Mutex Processing = new Mutex();

    int CurrentEpoch;
    bool TrainingDone;

    ManagedDNN Network = new ManagedDNN();
    NeuralNetworkOptions Options = new NeuralNetworkOptions();
    ManagedArray InputData = new ManagedArray();
    ManagedArray OutputData = new ManagedArray();
    ManagedArray TestData = new ManagedArray();
    ManagedArray NormalizationData = new ManagedArray();

    string FileName, FileNetwork;

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

        Idle.Add(new IdleHandler(OnIdle));
    }

    protected void ToggleControls(bool toggle)
    {
        FilenameTrainingData.Sensitive = toggle;
        ViewTrainingData.Sensitive = toggle;
        OpenTrainingDataButton.Sensitive = toggle;
        ReloadTrainingDataButton.Sensitive = toggle;

        FilenameTestData.Sensitive = toggle;
        ViewTestData.Sensitive = toggle;
        OpenTestDataButton.Sensitive = toggle;
        ReloadTestDataButton.Sensitive = toggle;

        InputLayerNodes.Sensitive = toggle;
        Categories.Sensitive = toggle;
        Examples.Sensitive = toggle;
        Samples.Sensitive = toggle;

        DelimiterBox.Sensitive = toggle;
    }

    protected void Pause()
    {
        if (Paused)
            return;

        Paused = true;

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

    protected void Run()
    {
        if (!Paused)
            return;

        Paused = false;

        ToggleControls(Paused);
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

        Processing.ReleaseMutex();

        return true;
    }

    protected void OnOpenTrainingDataButtonClicked(object sender, EventArgs e)
    {
        LoadTextFile(ref TrainingSetFileName, "Load Training Data", ViewTrainingData, FilenameTrainingData, true, Categories);

        UpdateParameters(ViewTrainingData, Examples, InputLayerNodes, true);

        //HiddenLayerNodes.Value = 2 * InputLayerNodes.Value;
    }

    protected void OnReloadTrainingDataButtonClicked(object sender, EventArgs e)
    {
        if (!string.IsNullOrEmpty(TrainingSetFileName))
            ReloadTextFile(TrainingSetFileName, ViewTrainingData, true, Categories);

        UpdateParameters(ViewTrainingData, Examples, InputLayerNodes, true);

        //HiddenLayerNodes.Value = 2 * InputLayerNodes.Value
    }

    protected void OnOpenTestDataButtonClicked(object sender, EventArgs e)
    {
        LoadTextFile(ref TestSetFileName, "Load Test Data", ViewTestData, FilenameTestData, false, null);

        UpdateParameters(ViewTestData, Samples, null, false);
    }

    protected void OnReloadTestDataButtonClicked(object sender, EventArgs e)
    {
    }

    protected void OnMainNotebookSwitchPage(object o, SwitchPageArgs args)
    {
    }
}
