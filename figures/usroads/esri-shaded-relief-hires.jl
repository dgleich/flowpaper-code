


#= get a hires esri shaded relief image.

We are going to download a static, hires shaded relief image from the ESRI servers.

By default basemap in python will call
http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/export?bbox=-2045969.4173062348,-2567274.0527822566,2546397.796126721,1048917.582910654&bboxSR=2163&imageSR=2163&size=5000,3937&dpi=96&format=png32&f=image
for our iamge.

But this only gets 4096 pixels in xpixels because of esri limits.  But we can tile ourselves...

We are going to tile for a 10k by Y image, where Y is computed so that
Y = floor(Int,10000*0.7874352088211344)

Then generate calls to download with chrome.

These will be named

ersi-shaded-relief-veryhires-1
ersi-shaded-relief-veryhires-2
...
ersi-shaded-relief-veryhires-6

as there are 6 of them.

To do this, we'll need to adjust the xmin and ymin


The standard call
#http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/export?bbox=-2045969.4173062348,-2567274.0527822566,2546397.796126721,1048917.582910654&bboxSR=2163&imageSR=2163&size=2600,4094&dpi=96&format=png32&f=image
=#


xmin,ymin,xmax,ymax = -2045969.4173062348,-2567274.0527822566,2546397.796126721,1048917.582910654
println("$xmin,$ymin,$(xmin + (xmax-xmin)/2),$ymax")
println("$(xmin + (xmax-xmin)/2),$ymin,$xmax,$ymax")

## Montage them
# http://www.imagemagick.org/Usage/montage/#montage
cmd = `montage esri-shaded-relief-veryhires-1.png esri-shaded-relief-veryhires-2.png -geometry +0+0 -tile 2x1 out.png`
run(cmd)

## This shows that everything works for a 2x1 tile. Now let's try our general tile.
aspect = 0.7874352088211344
xsize = 10005 # chosen so we get xsize/3 and ysize/2 are integers
ysize = floor(Int,xsize*aspect)
# we want stuff that's easy
@show xsize/3, ysize/2

xvals = range(xmin,xmax,length=4) # length = +1 because we want all the intermediate
yvals = range(ymin,ymax,length=3)
xtile = Int(xsize/3)
ytile = Int(ysize/2)

curimage = 1
for j=1:length(yvals)-1
  for i=1:length(xvals)-1
    global curimage
    url="http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/export?bbox=$(xvals[i]),$(yvals[j]),$(xvals[i+1]),$(yvals[j+1])&bboxSR=2163&imageSR=2163&size=$xtile,$ytile&dpi=96&format=png32&f=image"
    display(url)
    download(url, "ersi-shaded-relief-veryhires-$curimage.png")
    curimage += 1
  end
end

##
cmd = `montage ersi-shaded-relief-veryhires-4.png ersi-shaded-relief-veryhires-5.png ersi-shaded-relief-veryhires-6.png ersi-shaded-relief-veryhires-1.png ersi-shaded-relief-veryhires-2.png ersi-shaded-relief-veryhires-3.png  -geometry +0+0 -tile 3x2 esri-shaded-relief-veryhires.png`
run(cmd)



#= Old initial notes

We are going to setup code to cache the arcgis shaded relief output as this
can take a while to download and fails sometime.

When I used to use the bm.arcgisimage(...) call that is now commented out,
it would download the following picture.

http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/export?bbox=-2045969.4173062348,-2567274.0527822566,2546397.796126721,1048917.582910654&bboxSR=2163&imageSR=2163&size=5000,3937&dpi=96&format=png32&f=image

based on the code


"""
if not hasattr(self,'epsg'):
            msg = dedent("""
            Basemap instance must be creating using an EPSG code
            (http://spatialreference.org) in order to use the wmsmap method""")
            raise ValueError(msg)
        ax = kwargs.pop('ax', None) or self._check_ax()
        # find the x,y values at the corner points.
        p = pyproj.Proj(init="epsg:%s" % self.epsg, preserve_units=True)
        xmin,ymin = p(self.llcrnrlon,self.llcrnrlat)
        xmax,ymax = p(self.urcrnrlon,self.urcrnrlat)
        if self.projection in _cylproj:
            Dateline =\
            _geoslib.Point(self(180.,0.5*(self.llcrnrlat+self.urcrnrlat)))
            hasDateline = Dateline.within(self._boundarypolyxy)
            if hasDateline:
                msg=dedent("""
                arcgisimage cannot handle images that cross
                the dateline for cylindrical projections.""")
                raise ValueError(msg)
        if self.projection == 'cyl':
            xmin = (180./np.pi)*xmin; xmax = (180./np.pi)*xmax
            ymin = (180./np.pi)*ymin; ymax = (180./np.pi)*ymax
        # ypixels not given, find by scaling xpixels by the map aspect ratio.
        if ypixels is None:
            ypixels = int(self.aspect*xpixels)
        # construct a URL using the ArcGIS Server REST API.
        basemap_url = \
"%s/rest/services/%s/MapServer/export?\
bbox=%s,%s,%s,%s&\
bboxSR=%s&\
imageSR=%s&\
size=%s,%s&\
dpi=%s&\
format=png32&\
transparent=true&\
f=image" %\
(server,service,xmin,ymin,xmax,ymax,self.epsg,self.epsg,xpixels,ypixels,dpi)
        # print URL?
        if verbose: print(basemap_url)
        # return AxesImage instance.
        return self.imshow(imread(urlopen(basemap_url)),ax=ax,
                           origin='upper')
"""

http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/export?bbox=-2045969.4173062348,-2567274.0527822566,2546397.796126721,1048917.582910654&bboxSR=2163&imageSR=2163&size=5000,3937&dpi=96&format=png32&f=image

I wanted a 7500 pixel version, so I need the aspect ratio to adjust ypixels.

This involves finding the aspect to compute the ypixels, so I got the aspect

0.7874352088211344

http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/export?bbox=-2045969.4173062348,-2567274.0527822566,2546397.796126721,1048917.582910654&bboxSR=2163&imageSR=2163&size=7500,5905&dpi=96&format=png32&f=image



=#
