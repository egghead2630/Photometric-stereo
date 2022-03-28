import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy.sparse
image_row = 0 
image_col = 0
def Read_LightSource(Name):    
    path = './test/'
    path = path + Name + '/LightSource.txt'
    f = open(path,'r')
    L = []
    for line in f:
        line = line[6:]
        line = line[1:-2]
        x,y,z = line.split(',')
        x = int(x)
        y = int(y)
        z = int(z)
        unit = [x,y,z]
        L.append(unit)
    return L
def Read_Image(Name):
    path = './test/'
    _dir = path + Name + '/' 
    file_name = _dir + 'LightSource.txt'
    f = open(file_name,'r')

    I = []
    for line in f:
        pic_name = line[0:4]
        pic_name = pic_name+'.bmp'
        
        pic_name = _dir+pic_name
        #print(pic_name)
        p = cv2.imread(pic_name,cv2.IMREAD_GRAYSCALE)
        row_num = p.shape[0]
        col_num = p.shape[1]
        #p = p.flatten()
        p = np.reshape(p,(row_num * col_num))
        I.append(p)
    
    return I, (row_num,col_num)


# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image
def reconstruct(name):
    global image_row
    global image_col
    L = Read_LightSource(name)
    L = np.array(L)
    print(L.shape)
                    # get light direction L
    I, (image_row, image_col) = Read_Image(name)
    I = np.array(I)
    print(I.shape)
                    # get flattened image matrix I
    LTL = np.matmul(L.T,L)
    inv_LTL = np.linalg.pinv(LTL)
    LTI = np.matmul(L.T,I)

    N = np.matmul(inv_LTL,LTI)
    row_num = N.shape[0]
    col_num = N.shape[1]
    for i in range(col_num):
        square = N[0][i] * N[0][i] + N[1][i] * N[1][i] + N[2][i] * N[2][i]
        if square != 0:
            leng = np.sqrt(square)
            N[0][i] = N[0][i] / leng
            N[1][i] = N[1][i] / leng
            N[2][i] = N[2][i] / leng
            #square = N[0][i] * N[0][i] + N[1][i] * N[1][i] + N[2][i] * N[2][i]
            #leng = np.sqrt(square)
            #print(leng)
    print(N.shape)
    print(N)
                    # build normal map
    #N = N.T
    #normal_visualization(N)
    V = []            
    for i in range(col_num):
        nx = N[0][i]
        ny = N[1][i]
        nz = N[2][i]
        if nz != 0:
            V.append( -(nx/nz) )
            V.append( -(ny/nz) )
        else:
            V.append(0)
            V.append(0)
    V = np.array(V)
    #M = np.zeros( (2 * image_row * image_col, image_row * image_col) )
    M = scipy.sparse.lil_matrix((2 * image_row * image_col, image_row * image_col) )
    x = 0
    y = 0
    
    cnt = 0
    while cnt < 2 * image_row * image_col:
        if x + 1 < image_row:
            M[cnt, x + y * image_row] = -1
            M[cnt, x + 1 + y * image_row] = 1
            cnt += 1
        else:
            M[cnt, x + y * image_row] = 1
            M[cnt, x - 1 + y * image_row] = -1
            cnt += 1
        
        if y + 1 < image_col:
            M[cnt, x + y * image_row] = -1
            M[cnt, x + (y + 1) * image_row] = 1
            cnt += 1
        else:
            M[cnt, x + y * image_row] = 1
            M[cnt, x + (y - 1) * image_row] = -1
            cnt += 1


        if x + 1 == image_row:
            x = 0
            y += 1
        else:
            x += 1
    print(M.T.shape)
    print(M.shape)
    print('multiplying MTM')
    #MTM = np.matmul(M.T,M)
    MT = M.T
    MTM=MT.dot(M)
    print('Done')
#    print('solving MTM inverse')
#    inv_MTM = np.linalg.pinv(MTM)
#    print('Done')
    MTM = scipy.sparse.csr_matrix(MTM)
    print('multiplying MTV')
    #MTV = np.matmul(M.T,V)
    MTV=M.T.dot(V)
    print('Done')
    #MTV = scipy.sparse.csr_matrix(MTV)
    
    print('solving Z')
    Z = scipy.sparse.linalg.spsolve(MTM,MTV)
    print('Done')
   
#    print('Solving Z')
#    Z = np.matmul(inv_MTM,MTV)
#    print('Done')

    #print('Visualizing')
    #depth_visualization(Z)
    #normal_visualization(N.T)
    #print('Done')
                    # build Mz = v system
    # showing the windows of all visualization function
    save_ply(Z,name + '.ply')
    show_ply(name + '.ply')
 

if __name__ == '__main__':
    reconstruct('bunny')
    reconstruct('star')
    reconstruct('venus')
    plt.show()

