from paraview.simple import *


def render_csv_with_paraview(filename):
    # Renders the data in the given csv file using Paraview

    # disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'CSV Reader'
    data_from_csv = CSVReader(registrationName=filename.split('.')[0], FileName=[filename])

    # create a new 'Table To Points'
    tableToPoints1 = TableToPoints(registrationName='TableToPoints1', Input=data_from_csv)
    tableToPoints1.XColumn = 'u'
    tableToPoints1.YColumn = 'u'
    tableToPoints1.ZColumn = 'u'

    # Properties modified on tableToPoints1
    tableToPoints1.XColumn = 'x'
    tableToPoints1.YColumn = 'y'
    tableToPoints1.ZColumn = 'z'

    # show data in view
    renderView1 = FindViewOrCreate('RenderView1', viewtype='RenderView')
    tableToPoints1Display = Show(tableToPoints1, renderView1, 'GeometryRepresentation')
    layout1 = GetLayoutByName('Layout #1')

    # trace defaults for the display properties.
    tableToPoints1Display.Representation = 'Surface'
    tableToPoints1Display.ColorArrayName = [None, '']
    tableToPoints1Display.SelectTCoordArray = 'None'
    tableToPoints1Display.SelectNormalArray = 'None'
    tableToPoints1Display.SelectTangentArray = 'None'
    tableToPoints1Display.OSPRayScaleArray = 'u'
    tableToPoints1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    tableToPoints1Display.SelectOrientationVectors = 'None'
    tableToPoints1Display.ScaleFactor = 0.09500000000000001
    tableToPoints1Display.SelectScaleArray = 'None'
    tableToPoints1Display.GlyphType = 'Arrow'
    tableToPoints1Display.GlyphTableIndexArray = 'None'
    tableToPoints1Display.GaussianRadius = 0.004750000000000001
    tableToPoints1Display.SetScaleArray = ['POINTS', 'u']
    tableToPoints1Display.ScaleTransferFunction = 'PiecewiseFunction'
    tableToPoints1Display.OpacityArray = ['POINTS', 'u']
    tableToPoints1Display.OpacityTransferFunction = 'PiecewiseFunction'
    tableToPoints1Display.DataAxesGrid = 'GridAxesRepresentation'
    tableToPoints1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    tableToPoints1Display.ScaleTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    tableToPoints1Display.OpacityTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera(True)

    #changing interaction mode based on data extents
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.496042216358839, -0.5, 10000.0]
    renderView1.CameraFocalPoint = [0.496042216358839, -0.5, 0.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Delaunay 2D'
    delaunay2D1 = Delaunay2D(registrationName='Delaunay2D1', Input=tableToPoints1)

    # Properties modified on delaunay2D1
    delaunay2D1.ProjectionPlaneMode = 'Best-Fitting Plane'

    # show data in view
    delaunay2D1Display = Show(delaunay2D1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    delaunay2D1Display.Representation = 'Surface'
    delaunay2D1Display.ColorArrayName = [None, '']
    delaunay2D1Display.SelectTCoordArray = 'None'
    delaunay2D1Display.SelectNormalArray = 'None'
    delaunay2D1Display.SelectTangentArray = 'None'
    delaunay2D1Display.OSPRayScaleArray = 'u'
    delaunay2D1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    delaunay2D1Display.SelectOrientationVectors = 'None'
    delaunay2D1Display.ScaleFactor = 0.09500000000000001
    delaunay2D1Display.SelectScaleArray = 'None'
    delaunay2D1Display.GlyphType = 'Arrow'
    delaunay2D1Display.GlyphTableIndexArray = 'None'
    delaunay2D1Display.GaussianRadius = 0.004750000000000001
    delaunay2D1Display.SetScaleArray = ['POINTS', 'u']
    delaunay2D1Display.ScaleTransferFunction = 'PiecewiseFunction'
    delaunay2D1Display.OpacityArray = ['POINTS', 'u']
    delaunay2D1Display.OpacityTransferFunction = 'PiecewiseFunction'
    delaunay2D1Display.DataAxesGrid = 'GridAxesRepresentation'
    delaunay2D1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    delaunay2D1Display.ScaleTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    delaunay2D1Display.OpacityTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(tableToPoints1, renderView1)

    # update the view to ensure updated data information
    renderView1.Update()

    # set scalar coloring
    ColorBy(delaunay2D1Display, ('POINTS', 'zeta'))

    # rescale color and/or opacity maps used to include current data range
    delaunay2D1Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    delaunay2D1Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'zeta'
    zetaLUT = GetColorTransferFunction('zeta')

    # get opacity transfer function/opacity map for 'zeta'
    zetaPWF = GetOpacityTransferFunction('zeta')

    # set active source
    SetActiveSource(tableToPoints1)

    # set active source
    SetActiveSource(delaunay2D1)

    # create a new 'Calculator'
    calculator1 = Calculator(registrationName='Calculator1', Input=delaunay2D1)
    calculator1.Function = ''

    # Properties modified on calculator1
    calculator1.ResultArrayName = 'V'
    calculator1.Function = 'u*iHat + v*jHat'

    # show data in view
    calculator1Display = Show(calculator1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    calculator1Display.Representation = 'Surface'
    calculator1Display.ColorArrayName = ['POINTS', 'zeta']
    calculator1Display.LookupTable = zetaLUT
    calculator1Display.SelectTCoordArray = 'None'
    calculator1Display.SelectNormalArray = 'None'
    calculator1Display.SelectTangentArray = 'None'
    calculator1Display.OSPRayScaleArray = 'V'
    calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    calculator1Display.SelectOrientationVectors = 'V'
    calculator1Display.ScaleFactor = 0.09500000000000001
    calculator1Display.SelectScaleArray = 'None'
    calculator1Display.GlyphType = 'Arrow'
    calculator1Display.GlyphTableIndexArray = 'None'
    calculator1Display.GaussianRadius = 0.004750000000000001
    calculator1Display.SetScaleArray = ['POINTS', 'V']
    calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
    calculator1Display.OpacityArray = ['POINTS', 'V']
    calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
    calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
    calculator1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    calculator1Display.ScaleTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    calculator1Display.OpacityTransferFunction.Points = [-3.95778364116095, 0.0, 0.5, 0.0, 1.3192612137203161, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(delaunay2D1, renderView1)

    # show color bar/color legend
    calculator1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # Create arrow glyphs
    arrow_glyph = Glyph(registrationName='V arrows', Input=calculator1, GlyphType='2D Glyph')
    arrow_glyph.OrientationArray = ['POINTS', 'V']
    arrow_glyph.ScaleArray = ['POINTS', 'V']
    arrow_glyph.GlyphTransform = 'Transform2'
    arrow_glyph.ScaleFactor = 0.001

    # show data in view
    arrow_glyph_display = Show(arrow_glyph, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    arrow_glyph_display.Representation = 'Surface'
    arrow_glyph_display.ColorArrayName = ['POINTS', 'zeta']
    arrow_glyph_display.LookupTable = zetaLUT
    arrow_glyph_display.SelectTCoordArray = 'None'
    arrow_glyph_display.SelectNormalArray = 'None'
    arrow_glyph_display.SelectTangentArray = 'None'
    arrow_glyph_display.OSPRayScaleArray = 'V'
    arrow_glyph_display.OSPRayScaleFunction = 'PiecewiseFunction'
    arrow_glyph_display.SelectOrientationVectors = 'V'
    arrow_glyph_display.ScaleFactor = 0.10325857760035434
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

    #================================================================
    # addendum: following script captures some of the application
    # state to faithfully reproduce the visualization during playback
    #================================================================

    #--------------------------------
    # saving layout sizes for layouts

    # layout/tab size in pixels
    layout1.SetSize(1572, 809)

    #-----------------------------------
    # saving camera placements for views

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [0.496042216358839, -0.5, 10000.0]
    renderView1.CameraFocalPoint = [0.496042216358839, -0.5, 0.0]
    renderView1.CameraParallelScale = 0.5551279345612684

    #--------------------------------------------
    # uncomment the following to render all views
    Interact()
    #RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).
    SaveScreenshot("image.png")