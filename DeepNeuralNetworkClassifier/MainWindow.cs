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

    protected string GetBaseFileName(string fullpath)
    {
        return System.IO.Path.GetFileNameWithoutExtension(fullpath);
    }

    protected string GetDirectory(string fullpath)
    {
        return System.IO.Path.GetDirectoryName(fullpath);
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
    }

    protected void OnReloadTrainingDataButtonClicked(object sender, EventArgs e)
    {
    }

    protected void OnOpenTestDataButtonClicked(object sender, EventArgs e)
    {
    }

    protected void OnReloadTestDataButtonClicked(object sender, EventArgs e)
    {
    }
}
