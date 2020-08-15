#!/home/oxygen/anaconda3/envs/HCPpipeline/bin/python
import sys
#SurfPath=sys.argv[1]
#LabelPath=sys.argv[2]
#OutputFilePath=sys.argv[3]
FileRoot=sys.argv[1]
PatientID=sys.argv[2]



import gdist
import numpy as np
import nibabel as nib
import math
import networkx as nx
from networkx.algorithms import shortest_path,shortest_path_length
from itertools import combinations,chain
import pandas as pd
import pickle,os
class SurfaceDistance():
    def __init__(self,surface,labels,ResultPath=None):
        self.surface_path=surface
        self.labels_path=labels
        self.surf=nib.freesurfer.read_geometry(surface)
        self.cort = np.sort(nib.freesurfer.read_label(labels))
        self.ResultPath=ResultPath


    def run(self):
        VertsNeeded_set=set(self.cort)
        face_all=self.surf[1]
        verts_all=self.surf[0]
        ROIContainCount=self.counts_contain(Verts_Array=face_all,
                                            VertsNeeded_set=VertsNeeded_set)
        # find the face and edge
        face_use=self.find_SurfaceNeeded(face_all=face_all,ROIContainCount=ROIContainCount)
        self.face_use=face_use
        edge_verts=self.find_Edge(face_all=face_all,
                            verts_onROI_set=VertsNeeded_set,
                            ROIContainCount=ROIContainCount,
                            face_use=face_use)
        self.edge_verts=edge_verts

        # Maximum distance on the face_use
        MaxLength,MaxLength_sv,MaxLength_tv,_=self.find_Lengths(Target_Verts=edge_verts,
                                                                                   Source_Verts=edge_verts,
                                                                                   Faces=face_use,
                                                                                   Verts=verts_all,
                                                                                   FindMax=True).values()
        self.MaxLength=MaxLength
        self.MaxLength_sv,self.MaxLength_tv=MaxLength_sv,MaxLength_tv

        # Find MaxLength Path
        ConnectDf=self.build_ConnectionGraphDf(face_use=face_use,verts_all=verts_all)
        MaxLength_Verts=self.find_MaxLengthPath(ConnectDf,Source_Verts=MaxLength_sv,Target_Verts=MaxLength_tv)
        self.ConnectDf=ConnectDf
        self.MaxLength_Verts=MaxLength_Verts
        # Find Upper and Lower Edge based on the start and end point of MaxLength.
        Upper_Edge,Lower_Edge=self.find_SideOfEdge(ConnectDf,
                                                   Source_Verts=MaxLength_sv,
                                                   Target_Verts=MaxLength_tv,
                                                   Edge_Verts=edge_verts).values()
        self.Upper_Edge=Upper_Edge
        self.Lower_Edge=Lower_Edge

        # Maximum distance from both edge to the MaxLength Path (Max of the Minimum)
        MaxLengthUpper2Mid,MaxLengthUpper2Mid_sv,\
        MaxLengthUpper2Mid_tv,_= (self.find_Lengths(
                                                    Target_Verts=MaxLength_Verts,
                                                    Source_Verts=Upper_Edge,
                                                    Faces=face_use,
                                                    Verts=verts_all,
                                                    FindMax=False)
                                  .values())

        self.MaxLengthUpper2Mid=MaxLengthUpper2Mid
        self.MaxLengthUpper2Mid_sv,self.MaxLengthUpper2Mid_tv=MaxLengthUpper2Mid_sv,MaxLengthUpper2Mid_tv

        MaxLengthLower2Mid,MaxLengthLower2Mid_sv,\
        MaxLengthLower2Mid_tv,_= (self.find_Lengths(
                                                    Target_Verts=MaxLength_Verts,
                                                    Source_Verts=Lower_Edge,
                                                    Faces=face_use,
                                                    Verts=verts_all,
                                                    FindMax=False)
                                  .values())
        self.MaxLengthLower2Mid=MaxLengthLower2Mid
        self.MaxLengthLower2Mid_sv,self.MaxLengthLower2Mid_tv=MaxLengthLower2Mid_sv,MaxLengthLower2Mid_tv

        self.save_data()
        return MaxLength,MaxLengthLower2Mid,MaxLengthUpper2Mid,self.ResultPath



    def Length(self,vertices,triangles,source_indices,target_indices,FindMax=True):
        RawArray=gdist.compute_gdist(vertices, triangles,
                                     source_indices,target_indices)
        cleaned=RawArray[~np.isinf(RawArray)]
        source=source_indices[0]
        if FindMax:
            func=max
        else:
            func=min
        if len(cleaned)!=0:
            V=func(cleaned)
            target=target_indices[RawArray==V][0]
            # maybe a bug!
            # if have two points on the edge that share the same and max length, use the first one.

        else:

            V=0
            target=source

        return {'ShorestDist':V,
                'SourceIndex':source,
                'TargetIndex':target}

    def distance(self,p1, p2):
        x1, y1, z1=p1
        x2, y2, z2=p2
        d = math.sqrt(math.pow(x2 - x1, 2) +
                    math.pow(y2 - y1, 2) +
                    math.pow(z2 - z1, 2)* 1.0)
        return d

    def counts_contain(self,Verts_Array,VertsNeeded_set):

        N=np.array(list(map(lambda x:len(set(x).intersection(VertsNeeded_set))
                            ,Verts_Array)))
        return N

    def find_Edge(self,face_all,verts_onROI_set,ROIContainCount,face_use):
        Edge0=set(list(set(face_all[(ROIContainCount==2)|(ROIContainCount==1)].flatten())
                           .intersection(verts_onROI_set)))
        # edge was defined as only 2 verts of the face within the verts_onROI_set
        # however Maybe bug!!

        # as under this definition some verts of edge do not on
        # the surface_use(all 3 verts within the verts_onROI_set)

        Edge=np.array(list(set(face_use.flatten()).intersection(Edge0)),dtype=np.int32)
        return Edge

    def find_SurfaceNeeded(self,face_all,ROIContainCount):
        face_use=face_all[ROIContainCount!=0].astype(np.int32)
        return face_use


    def find_Lengths(self,Target_Verts,Source_Verts,Faces,Verts,FindMax=True):
        # FindMax:
        # True:find the Longest amount the shorest distances
        # False: find the Shortest amount the shorest distances
        LengthBetweenEdge=pd.DataFrame(list(map(lambda x: self.Length(Verts,
                                                                      Faces,
                                                                      source_indices = np.array([x],dtype=np.int32),
                                                                      target_indices=Target_Verts,
                                                                      FindMax=FindMax),
                                                Source_Verts)))

        MaxLength,source_Vert,target_Vert=(LengthBetweenEdge[LengthBetweenEdge.ShorestDist==
                                                            LengthBetweenEdge.ShorestDist.max()]
                                           .values[0])


        return {'MaxLength':MaxLength,
                'MaxLength_source':source_Vert,
                'MaxLength_target':target_Vert,
                'Df':LengthBetweenEdge}

    def build_ConnectionGraphDf(self,face_use,verts_all):
        VertsCombinations=list(map(lambda x:tuple(combinations(x,2)),face_use))
        VertsCombinations_undup=list(set(list(chain(*VertsCombinations))))
        # distance between VertsCombinations
        VV_distance=list(map(lambda x:self.distance(verts_all[x[0]],verts_all[x[1]]),
                           VertsCombinations_undup))

        ConnectDf=(pd.DataFrame(VertsCombinations_undup,VV_distance)
                   .reset_index()
                   .rename(columns={'index':'Weight',0:'source',1:'target'}))
        return ConnectDf



    def find_MaxLengthPath(self,ConnectDf,Source_Verts,Target_Verts):
        g=nx.from_pandas_edgelist(ConnectDf,'source', 'target', ['Weight'])
        g_use=g.to_undirected()
        # max length path on the surface,
        # note:
        # the length of the path based on shortest path only is longer
        # than the distance generated by gdist.compute_gdist
        # maybe a bug!
        ShortestPath_Verts=shortest_path(G=g_use,source=Source_Verts,target=Target_Verts,weight='Weight')
        ShortestPath_Verts=np.array(ShortestPath_Verts,dtype=np.int32)
        return ShortestPath_Verts

    def find_SideOfEdge(self,ConnectDf,Source_Verts,Target_Verts,Edge_Verts):
        edge_set=set(Edge_Verts)
        # build a net only contain the edge
        onEdge_N=self.counts_contain(Verts_Array=ConnectDf.values,VertsNeeded_set=edge_set)
        g_edge=nx.from_pandas_edgelist(ConnectDf[onEdge_N==2],'source',
                                       'target', ['Weight']).to_undirected()
        # edge on the one side of the max length path
        ShortestPath_OnEdge1=shortest_path(G=g_edge,source=Source_Verts,target=Target_Verts,weight='Weight')

        # build a edge net without the Verts on the ShortestPath_OnEdge1
        edgeSet_shortestRm=(edge_set-set(ShortestPath_OnEdge1)).union(set([Source_Verts,Target_Verts]))
        onEdge_N1=self.counts_contain(Verts_Array=ConnectDf.values,VertsNeeded_set=edgeSet_shortestRm)
        g_edge1=nx.from_pandas_edgelist(ConnectDf[onEdge_N1==2],
                                        'source', 'target', ['Weight']).to_undirected()
        # edge on the other side of the max length path
        try:
            ShortestPath_OnEdge2=shortest_path(G=g_edge1,source=Source_Verts,target=Target_Verts,weight='Weight')
        except:
            ShortestPath_OnEdge2=list(edgeSet_shortestRm)
            print('Edge incomplete, maybe not a closed circle!')

        ShortestPath_OnEdge1=np.array(ShortestPath_OnEdge1,dtype=np.int32)
        ShortestPath_OnEdge2=np.array(ShortestPath_OnEdge2,dtype=np.int32)
        return {'OneSideOfEdge_Verts':ShortestPath_OnEdge1,
                'OtherSiderOfEdge_Verts':ShortestPath_OnEdge2}


    def save_data(self):
        Path=self.ResultPath
        if Path!=None:
            root=os.path.dirname(Path)
            BasicInfo={'SurfacePath':self.surface_path,'LabelPath':self.labels_path,
                        'face_use':self.face_use,'edge_use':self.edge_verts,
                        'ConnectDf':self.ConnectDf}
            Result={'MaxLength':self.MaxLength,'MaxHight':self.MaxLengthUpper2Mid+self.MaxLengthLower2Mid,
                    'MaxLength_SourceVect':self.MaxLength_sv,'MaxLength_TargetVect':self.MaxLength_tv,
                    'UpperEdge':self.Upper_Edge,'LowerEdge':self.Lower_Edge,
                    'Lower2Mid':self.MaxLengthLower2Mid,
                    'Lower2Mid_SourceVect':self.MaxLengthLower2Mid_sv,'Lower2Mid_TargetVect':self.MaxLengthLower2Mid_tv,
                    'Upper2Mid':self.MaxLengthUpper2Mid,
                    'Upper2Mid_SourceVect':self.MaxLengthUpper2Mid_sv,'Upper2Mid_TargetVect':self.MaxLengthUpper2Mid_tv,}
            Result.update(BasicInfo)
            if not os.path.exists(root):
                os.makedirs(root)

            with open(Path,'wb') as f:
                pickle.dump(Result,f)
#            print('Result Saved: '+Path)
i=PatientID
for h in ['lh','rh']:
    surf=os.path.join(FileRoot,i,'T1w',i,'surf','.'.join([h,'pial']))
    label=os.path.join(FileRoot,i,'T1w',i,'label','.'.join([h,'V1_exvivo','label']))
    if os.path.exists(surf) and os.path.exists(label):
        length=SurfaceDistance(surf,label,ResultPath=label+'.updated.pickle')
        L,S1,S2,ResultFilePath=length.run()
        print(i,h,L,S1+S2,ResultFilePath)
