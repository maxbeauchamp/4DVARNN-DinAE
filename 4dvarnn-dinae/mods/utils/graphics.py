import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def plot(ax,i,j,lon,lat,data,title,extent=[-65,-55,30,40],cmap="coolwarm",gridded=True,vmin=-2,vmax=2):
    ax[i][j].set_extent(list(extent))
    if gridded:
        im=ax[i][j].pcolormesh(lon, lat, data, cmap=cmap,\
                          vmin=vmin, vmax=vmax,edgecolors='face', alpha=1, \
                          transform= ccrs.PlateCarree(central_longitude=0.0))
    else:
        im=ax[i][j].scatter(lon, lat, c=data, cmap=cmap, s=1,\
                       vmin=vmin, vmax=vmax,edgecolors='face', alpha=1, \
                       transform= ccrs.PlateCarree(central_longitude=0.0)) 
    im.set_clim(vmin,vmax)
    clb = plt.colorbar(im, orientation="horizontal", extend='both', pad=0.1, ax=ax[i][j])
    ax[i][j].set_title(title, pad=40, fontsize = 15)
    gl = ax[i][j].gridlines(alpha=0.5,draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_bottom = False
    gl.ylabels_right = False
    gl.xlabel_style = {'fontsize': 10, 'rotation' : 45}
    gl.ylabel_style = {'fontsize': 10}
    ax[i][j].coastlines(resolution='50m')
