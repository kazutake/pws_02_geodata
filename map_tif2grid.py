from numba import njit, jit, f8
import time
import gdal, gdalconst
import h5py
import numpy as np
import yaml
import subprocess
from subprocess import PIPE
import os

def start_func(func_name):
    print('<-- ')
    print('start: ', func_name)
    return time.time()

def end_func(func_name, start_time):
    elapsed_time = time.time() - start_time
    print('end: ', func_name)
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print (" -->")

    return 0

# output csv
def output_csv(config, imax, jmax, xx, yy, zb, sn, vege_d, vege_h):

    # csv 出力
    with open(config['cgns_file'] + "_grid.csv", mode='w') as f:
        l = ['imax,jmax,kmax,\n']
        f.writelines(l)
        l = [str(imax), ',' ,str(jmax), ',1,\n']
        f.writelines(l)
        l = ['i,j,k,x,y,z,N_Elevation,N_Elevation_zb,C_Obstacle,C_Fix_movable,C_vege_density,C_vege_height,C_roughness_cell,C_mix_cell\n']
        f.writelines(l)
        for j in range(jmax):
                for i in range(imax):

                    if i < imax-1 and j < jmax-1:
                        l = [str(i), ',' ,str(j), ',0,', \
                            str(xx[j,i]), ',' ,str(yy[j,i]), ',0,', \
                            str(zb[j,i]), ',-9999, 0, 0,', \
                            str(vege_d[j,i]), ',', str(vege_h[j,i]), ',', \
                            str(sn[j,i]), \
                            ', 0\n']
                    else:
                        l = [str(i), ',' ,str(j), ',0,', \
                            str(xx[j,i]), ',' ,str(yy[j,i]), ',0,', \
                            str(zb[j,i]), ',-9999, 0, 0, 0, 0, 0.025, 0\n']

                    f.writelines(l)

#　update cgns file
def update_grid(config, zb, sn, vege_d, vege_h):

    #open
    with h5py.File(config['cgns_file'], mode='r+') as f:
        dname = 'iRIC/iRICZone/GridConditions/Elevation/Value/ data'
        f[dname][()] = zb.reshape(-1)

        dname = 'iRIC/iRICZone/GridConditions/roughness_cell/Value/ data'
        f[dname][()] = sn.reshape(-1)

        dname = 'iRIC/iRICZone/GridConditions/vege_density/Value/ data'
        f[dname][()] = vege_d.reshape(-1)

        dname = 'iRIC/iRICZone/GridConditions/vege_height/Value/ data'
        f[dname][()] = vege_h.reshape(-1)


    return 0

# read the grid x, y, z values and return them
def read_grid(config):

    #open
    with h5py.File(config['cgns_file'], mode='r') as f:

        # grid information
        dname = 'iRIC/iRICZone/GridCoordinates/CoordinateX/ data'
        xx = f[dname][()]
        dname = 'iRIC/iRICZone/GridCoordinates/CoordinateY/ data'
        yy = f[dname][()]
        imax = xx.shape[1]
        jmax = xx.shape[0]

        # node values
        zz = np.zeros((jmax, imax), dtype=float)
        sn = np.zeros((jmax-1, imax-1), dtype=float)
        ivege = np.zeros((jmax-1, imax-1), dtype=int)

    return imax, jmax, xx, yy, zz, sn, ivege

# read the geotif data
def read_geotif(fname):

    # open
    src = gdal.Open(fname, gdalconst.GA_ReadOnly) 

    # set parameters
    n_cols = src.RasterXSize
    n_rows = src.RasterYSize
    profile = src.GetGeoTransform()
    dx = profile[1]; dy = - profile[5]
    xll = profile[0]; yll = profile[3] - dy * n_rows
    nodata = src.GetRasterBand(1).GetNoDataValue()
    vals = src.GetRasterBand(1).ReadAsArray()

    # output profile
    # print('n_cols:', n_cols, ', n_rows:', n_rows)
    # print('dx:', dx, ', dy:', dy)
    # print('xll:', xll, ', yll:', yll)

    return n_cols, n_rows, dx, dy, xll, yll, vals

@jit("int32[:,:](int64, int64, float64[:,:], float64[:,:], int32[:,:], int64, int64, float64, float64, float64, float64, int32[:,:])", nopython=True)
def map_raster_int_cell(imax, jmax, xx, yy, iv, n_cols, n_rows, dx, dy, xll, yll, vals):

    for j in range(jmax-1):
        for i in range(imax-1):
            xp = 0.25 * (xx[j,i] + xx[j+1,i] + xx[j,i+1] + xx[j+1,i+1])
            yp = 0.25 * (yy[j,i] + yy[j+1,i] + yy[j,i+1] + yy[j+1,i+1])
            col = int((xp - xll) / dx)
            row = n_rows - int((yp - yll) / dy) -1
            iv[j,i] = vals[row, col]
    return iv

@jit("int32[:,:](int64, int64, float64[:,:], float64[:,:], int32[:,:], int64, int64, float64, float64, float64, float64, int32[:,:])", nopython=True)
def map_raster_int_node(imax, jmax, xx, yy, iv, n_cols, n_rows, dx, dy, xll, yll, vals):

    for j in range(jmax):
        for i in range(imax):
            xp = xx[j,i]
            yp = yy[j,i]
            col = int((xp - xll) / dx)
            row = n_rows - int((yp - yll) / dy) -1
            iv[j,i] = vals[row, col]
    return iv

@jit("float64[:,:](int64, int64, float64[:,:], float64[:,:], float64[:,:], int64, int64, float64, float64, float64, float64, float32[:,:], int64, int64)", nopython=True)
def map_raster_real_cell(imax, jmax, xx, yy, vv, n_cols, n_rows, dx, dy, xll, yll, vals, ncell, method):

    for j in range(jmax-1):
        for i in range(imax-1):
            xp = 0.25 * (xx[j,i] + xx[j+1,i] + xx[j,i+1] + xx[j+1,i+1])
            yp = 0.25 * (yy[j,i] + yy[j+1,i] + yy[j,i+1] + yy[j+1,i+1])
            col = int((xp - xll) / dx)
            row = n_rows - int((yp - yll) / dy) -1

            zlist =[]
            for c in range(-ncell, ncell + 1):
                for r in range(-ncell, ncell + 1):
                    ir = row + r
                    ic = col + c
                    if (ir > 0 and ir < n_rows) and (ic > 0 and ic < n_cols) :
                        zlist.append(vals[ir, ic])

            if method == 0:
                ss = 0; count = 0
                for item in zlist:
                    ss = ss + item
                    count = count + 1

                vv[j,i] = ss/count
            
            elif method == 1:
                ss = -9999
                for item in zlist:
                    if item > ss:
                        ss = item
                vv[j,i] = ss

            elif method == 2:
                ss = 9999
                for item in zlist:
                    if item < ss:
                        ss = item
                vv[j,i] = ss

    return vv

@jit("float64[:,:](int64, int64, float64[:,:], float64[:,:], float64[:,:], int64, int64, float64, float64, float64, float64, float32[:,:], int64, int64)", nopython=True)
def map_raster_real_node(imax, jmax, xx, yy, vv, n_cols, n_rows, dx, dy, xll, yll, vals, ncell, method):

    for j in range(jmax):
        for i in range(imax):
            col = int((xx[j, i] - xll) / dx)
            row = n_rows - int((yy[j, i] - yll) / dy) -1

            zlist =[]
            for c in range(-ncell, ncell + 1):
                for r in range(-ncell, ncell + 1):
                    ir = row + r
                    ic = col + c
                    if (ir > 0 and ir < n_rows) and (ic > 0 and ic < n_cols) :
                        zlist.append(vals[ir, ic])

            # 以下はjitでは使えない
            # if method == 0:
            #     zz[j,i] = sum(zlist) / len(zlist)
            
            # elif method == 1:
            #     zz[j,i] = max(zlist)

            # elif method == 2:
            #     zz[j,i] = min(zlist)

            if method == 0:
                ss = 0; count = 0
                for item in zlist:
                    ss = ss + item
                    count = count + 1

                vv[j,i] = ss/count
            
            elif method == 1:
                ss = -9999
                for item in zlist:
                    if item > ss:
                        ss = item
                vv[j,i] = ss

            elif method == 2:
                ss = 9999
                for item in zlist:
                    if item < ss:
                        ss = item
                vv[j,i] = ss

    return vv

# make calc env
def set_out_dir(config):
    import os

    #出力ディレクトリがない場合は作成する
    base_dir = os.path.dirname(config['cgns_file'])
    out_dir = os.path.join(base_dir, 'img')
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)

    return out_dir

# main function
def main(args):
    

    # read condition file
    with open(args[1], 'r') as yml:
        config = yaml.load(yml)

    # set vegetation type
    vege = config['vege_param']

    #　計算格子の読み込み
    imax, jmax, xx, yy, zb0, sn0, ivege0 = read_grid(config)

    #　地形データtifの読み込み
    n_cols, n_rows, dx, dy, xll, yll, zb_data = read_geotif(config['tif_zb'])
    zb = map_raster_real_node(imax, jmax, xx, yy, zb0, \
                            n_cols, n_rows, dx, dy, xll, yll, zb_data, \
                            config['ncell'], config['method'])

    # 粗度
    n_cols, n_rows, dx, dy, xll, yll, sn_data = read_geotif(config['tif_sn'])
    sn = map_raster_real_cell(imax, jmax, xx, yy, sn0, \
                        n_cols, n_rows, dx, dy, xll, yll, sn_data, 
                        config['ncell'], config['method'])
    sn[:,:] = 0.02

    # 樹木
    n_cols, n_rows, dx, dy, xll, yll, ivege_data = read_geotif(config['tif_vege'])
    
    # これも本来は不要
    ivege_data = ivege_data.astype(np.int32)


    ivege = map_raster_int_cell(imax, jmax, xx, yy, ivege0, 
                            n_cols, n_rows, dx, dy, xll, yll, ivege_data)

    # 仮データしかないので、ゼロにする
    ivege = np.zeros_like(ivege)  # すべて0で初期化する

    vege_d = np.zeros((jmax-1, imax-1), dtype=float)
    vege_h = np.zeros((jmax-1, imax-1), dtype=float)
    for j in range(jmax-1):
        for i in range(imax-1):
            vege_d[j,i] = vege[ivege[j,i]][0]
            vege_h[j,i] = vege[ivege[j,i]][1]

    #update
    ier = update_grid(config, zb, sn, vege_d, vege_h)

    # output csv
    ier = output_csv(config, imax, jmax, xx, yy, zb, sn, vege_d, vege_h)


    return 0

#root
if __name__ == "__main__":
    import sys

    # start
    start_time = start_func('__main__')

    # 引数を取得しmainに
    args = sys.argv
    main(args)

    # end
    end_func('__main__', start_time)

    print('*** finish. ***')