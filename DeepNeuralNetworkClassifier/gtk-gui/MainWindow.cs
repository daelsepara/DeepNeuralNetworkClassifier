
// This file has been generated by the GUI designer. Do not modify.

public partial class MainWindow
{
	protected virtual void Build()
	{
		global::Stetic.Gui.Initialize(this);
		// Widget MainWindow
		this.WidthRequest = 800;
		this.HeightRequest = 600;
		this.Name = "MainWindow";
		this.Title = global::Mono.Unix.Catalog.GetString("MainWindow");
		this.WindowPosition = ((global::Gtk.WindowPosition)(4));
		this.Resizable = false;
		this.DefaultWidth = 800;
		this.DefaultHeight = 600;
		if ((this.Child != null))
		{
			this.Child.ShowAll();
		}
		this.Show();
		this.DeleteEvent += new global::Gtk.DeleteEventHandler(this.OnDeleteEvent);
	}
}
