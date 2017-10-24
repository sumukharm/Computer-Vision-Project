# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("[Input_Marking]")
   print("Path to the input marking")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=30)
    #print 'segments : ',len(segments)
    segments_ids = np.unique(segments)
	
    #print 'ids : ',len(segments_ids)
    #print '\n\nsegments :',len(segments)

    #print '-----------------------------------'
    #print segments
    #for i in xrange(0,len(segments)):
#	print segments[i]
    #print segments_ids
    #print '-----------------------------------'

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])
    #print len(colors_hists)


    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)    

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1/2.0)

        return total_diff;

if __name__ == '__main__':
   
    # validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img_marking = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # ======================================== #
    # write all your codes here
    centers,colors_hists,segments,tri=superpixels_histograms_neighbors(img)   
    #print 'segments : ',segments,len(segments)
    fg,bg=find_superpixels_under_marking(img_marking,segments)        
    #print len(fg),len(bg)
    #print segments[fg[0]]
    #print segments[fg[1]]
    #print 'centers : ',len(centers),'segments : ',len(segments),'color : ',len(colors_hists)
    
    #mask = cv2.cvtColor(img_marking, cv2.COLOR_RGB2GRAY) # dummy assignment for mask, change it to your result

    #count=0
    segments_ids=np.unique(segments)
    mask=np.zeros(img.shape[:2],dtype='uint8')
    #for (i,seg_val) in enumerate(bg):
    	#mask=np.zeros(img.shape[:2],dtype='uint8')
#	mask[segments==seg_val]=255

#	path='/home/srajapurammo/assignment4/trash/'+str(count)+'.png'
#	cv2.imwrite(path,cv2.bitwise_and(img,img,mask=mask))
#	count=count+1

    #cv2.imwrite('pix.png',pixels_for_segment_selection(fg,fg[2:3]))
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    #fg_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in fg])
    #bg_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in bg])
    
    fg_hists=cumulative_histogram_for_superpixels(fg,colors_hists)
    #fg_hists=normalize_histograms(fg_hists)
    bg_hists=cumulative_histogram_for_superpixels(bg,colors_hists)
    #bg_hists=normalize_histograms(bg_hists)
    #print len(fg_hists),len(bg_hists)    

    norm_hists=normalize_histograms(colors_hists)

    #print "-------------------------------------------------------------------------------------------------------------------------------------"
    #print 'norm_hist : ',norm_hists
    #print 'fg : ',fg
    #print 'bg : ',bg
    #fgbg_superpixels=fg+bg[:71]
    fgbg_superpixels=[fg,bg]
    #print 'zip : ',zip(fg,bg)
    #print '**********************************'
    #print 'fg_his : ',fg_hists
    #print 'bg_his : ',bg_hists
    #fgbg_hists=fg_hists+bg_hists 
    fgbg_hists=[fg_hists,bg_hists]
    #print 'zip : ',zip(fg_hists,bg_hists) 

    # ======================================== #

    graph_cut=do_graph_cut(fgbg_hists,fgbg_superpixels,norm_hists,tri)
    #print "............................................."
    #print graph_cut,len(graph_cut),len(segments_ids)
    #print "............................................."

    mask = cv2.cvtColor(img_marking, cv2.COLOR_RGB2GRAY) # dummy assignment for mask, change it to your result
    #count=0
    #segments_ids=np.unique(segments)
    mask=np.zeros(img.shape[:2],dtype='uint8')
    pair=zip(segments_ids,graph_cut)
    #print pair
    for (i,seg_val) in enumerate(pair):
    	#mask=np.zeros(img.shape[:2],dtype='uint8')
	if seg_val[1]==True:
        	mask[segments==seg_val[0]]=255
        elif seg_val[1]==False:
		mask[segments==seg_val[0]]=0	
	
	#path='/home/srajapurammo/assignment4/trash/'+str(count)+'.png'
	#cv2.imwrite(path,cv2.bitwise_and(img,img,mask=mask))
	#count=count+1

    img=cv2.imread('example_output.png',cv2.IMREAD_COLOR)
    
    img_master=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #mask_target=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    #print img.shape,mask.shape

    print 'RMSD : ',RMSD(mask,img_master)

    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('mask.png',mask)
    # read video file
    output_name = sys.argv[3] + "mask.png"
    #cv2.imshow('figure',mask)
    #cv2.waitKey(0)
    cv2.imwrite(output_name, mask);
