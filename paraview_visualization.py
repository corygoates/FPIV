from paraview.simple import *


def render_csv_with_paraview(filename, **kwargs):
    """Renders the given CSV data using paraview.
    
    Parameters
    ----------
    filename : str
        CSV file to read in the data from.

    image_file : str, optional
        File to write the rendered data to. Defaults to no output.

    arrow_scale_factor : float, optional
        Velocity vector length scaling. Defaults to 0.001.
    """

    # disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # Read in data from CSV
    data_from_csv = CSVReader(registrationName=filename.split('.')[0], FileName=[filename])

    # Apply table to points filter
    tableToPoints1 = TableToPoints(registrationName='TableToPoints1',
                                   Input=data_from_csv,
                                   XColumn='x',
                                   YColumn='y',
                                   ZColumn='z')

    # show data in view
    renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
    layout1 = GetLayoutByName('Layout #1')

    # create a new 'Delaunay 2D'
    delaunay2D1 = Delaunay2D(registrationName='Delaunay2D1', Input=tableToPoints1)

    # Properties modified on delaunay2D1
    delaunay2D1.ProjectionPlaneMode = 'Best-Fitting Plane'

    # show data in view
    delaunay2D1Display = Show(delaunay2D1, renderView1, 'GeometryRepresentation')

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    delaunay2D1Display.ScaleTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    delaunay2D1Display.OpacityTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # update the view to ensure updated data information
    renderView1.Update()

    # set active source
    SetActiveSource(delaunay2D1)

    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='Calculator1', Input=delaunay2D1, ResultArrayName='V', Function="u*iHat + v*jHat")

    # show data in view
    calculator1Display = Show(calculator1, renderView1, 'GeometryRepresentation')
    ColorBy(calculator1Display, ('POINTS', 'raw_data_1'))

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    calculator1Display.ScaleTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    calculator1Display.OpacityTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(delaunay2D1, renderView1)

    # update the view to ensure updated data information
    renderView1.Update()

    # Create arrow glyphs
    arrow_glyph = Glyph(registrationName='V arrows',
                        Input=calculator1,
                        GlyphType='2D Glyph',
                        ScaleFactor=kwargs.get("arrow_scale_factor", 0.001),
                        OrientationArray = ['POINTS', 'V'],
                        ScaleArray = ['POINTS', 'V'],
                        GlyphTransform = 'Transform2')

    # show data in view
    arrow_glyph_display = Show(arrow_glyph, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    arrow_glyph_display.Representation = 'Surface'
    arrow_glyph_display.ColorArrayName = ['POINTS', 'Solid Color']
    ColorBy(arrow_glyph_display, None)
    arrow_glyph_display.AmbientColor = [0.0, 0.0, 0.0]
    arrow_glyph_display.DiffuseColor = [0.0, 0.0, 0.0]
    arrow_glyph_display.SelectTCoordArray = 'None'
    arrow_glyph_display.SelectNormalArray = 'None'
    arrow_glyph_display.SelectTangentArray = 'None'
    arrow_glyph_display.OSPRayScaleArray = 'V'
    arrow_glyph_display.OSPRayScaleFunction = 'PiecewiseFunction'
    arrow_glyph_display.SelectOrientationVectors = 'V'
    arrow_glyph_display.SelectScaleArray = 'None'
    arrow_glyph_display.GlyphType = 'Arrow'
    arrow_glyph_display.GlyphTableIndexArray = 'None'
    arrow_glyph_display.GaussianRadius = 0.0051629288800177165
    arrow_glyph_display.SetScaleArray = ['POINTS', 'V']
    arrow_glyph_display.ScaleTransferFunction = 'PiecewiseFunction'
    arrow_glyph_display.OpacityArray = ['POINTS', 'V']
    arrow_glyph_display.OpacityTransferFunction = 'PiecewiseFunction'
    arrow_glyph_display.DataAxesGrid = 'GridAxesRepresentation'
    arrow_glyph_display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    arrow_glyph_display.ScaleTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    arrow_glyph_display.OpacityTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # show color bar/color legend
    arrow_glyph_display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # layout/tab size in pixels
    layout1.SetSize(800, 800)

    #-----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.OrientationAxesVisibility = 0
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.5, 0.5, 10000.0]
    renderView1.CameraFocalPoint = [0.5, 0.5, 0.0]
    renderView1.CameraParallelScale = 0.5551279345612684

    # Render and allow interaction
    Interact()

    # Save image
    image_name = kwargs.get("image_name")
    if image_name is not None:
        SaveScreenshot(image_name)