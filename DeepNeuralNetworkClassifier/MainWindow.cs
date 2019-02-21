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

    CultureInfo ci = new CultureInfo("en-us");

    public MainWindow() : base(Gtk.WindowType.Toplevel)
    {
        Build();

        InitializeUserInterface();
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

    protected void OnDeleteEvent(object sender, DeleteEventArgs a)
    {
        if (GetConfirmation())
        {
            Quit();
        }

        a.RetVal = true;
    }
}
