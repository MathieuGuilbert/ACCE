import matplotlib.pyplot as plt
import numpy as np

def vizualizeAnchors(anchors,data,partition,resPath):
    plt.clf()
    #print(anchors)
    h=len(anchors)
    idAnchor=-1
    anchors_x=[a[0] for a in anchors]
    anchors_y=[a[1] for a in anchors]
    n=len(data)

    #colors=['lightgreen','orange','blue','red','lightblue','brown','purple','green','black','white','yellow','indigo','pink','gray','gold']
    colors=['#E52B50','#FFBF00','#9966CC','#FBCEB1','#7FFFD4','#007FFF','#89CFF0','#F5F5DC','#CB4154','#000000','#0095B6','#8A2BE2','#DE5D83','#CD7F32','#964B00','#800020','#702963','#DE3163',
    '#007BA7','#F7E7CE','#7FFF00','#7B3F00','#0047AB','#6F4E37','#FF7F50','#DC143C','#EDC9AF','#50C878','#00FF3F','#FFD700','#808080','#008000','#4B0082','#B57EDC','#FFF700','#C8A2C8','#FF00FF',
    '#FF00AF','#800000','#E0B0FF','#000080','#CC7722','#808000','#FF6600','#FF4500','#DA70D6','#FFE5B4','#CCCCFF','#003153','#CC8899','#E30B5C','#FF0000','#C71585','#FF007F','#E0115F',
    '#92000A','#0F52BA','#C0C0C0','#708090','#A7FC00','#00FF7F','#D2B48C','#483C32','#008080','#40E0D0','#3F00FF','#7F00FF','#40826D','#FFFF00']

    for i in range(max(partition)+1):
        #print("Anchors "+str(i)+" out of "+str(h))
        if(i in partition):
            idAnchor+=1
            i_data=[data[p] for p in range(n) if partition[p] == i]
            i_data_x=[a[0] for a in i_data]
            i_data_y=[a[1] for a in i_data]
            #print(i_data)
            #plt.scatter(data[partition == i][0], data[partition == i][1],s=50, c=colors[i], marker='s', edgecolor='black',label='cluster'+str(i))
            plt.scatter(i_data_x, i_data_y, s=50, c=colors[i], marker='o', edgecolor='black',label='cluster'+str(i))
            #plt.scatter(anchors[partition == i][0], anchors[partition == i][1], marker='^', c=colors[i], s=70)
            plt.scatter([anchors_x[idAnchor]], [anchors_y[idAnchor]], marker='^', c=colors[i], s=170, edgecolor='black')
            #plt.scatter(cen_x, cen_y, marker='^', c=colors[i], s=70)

    plt.savefig(resPath)
    #plt.savefig(r"C:\Users\mguil\OneDrive\Bureau\test.png")

def vizualizePartition(k,data,partition,resPath):
    plt.clf()
    n=len(data)
    #colors=['lightgreen','orange','blue','red','lightblue','brown','purple','green','black','white','yellow','indigo','pink','gray','gold']
    colors=['#E52B50','#FFBF00','#9966CC','#FBCEB1','#7FFFD4','#007FFF','#89CFF0','#F5F5DC','#CB4154','#000000','#0095B6','#8A2BE2','#DE5D83','#CD7F32','#964B00','#800020','#702963','#DE3163',
    '#007BA7','#F7E7CE','#7FFF00','#7B3F00','#0047AB','#6F4E37','#FF7F50','#DC143C','#EDC9AF','#50C878','#00FF3F','#FFD700','#808080','#008000','#4B0082','#B57EDC','#FFF700','#C8A2C8','#FF00FF',
    '#FF00AF','#800000','#E0B0FF','#000080','#CC7722','#808000','#FF6600','#FF4500','#DA70D6','#FFE5B4','#CCCCFF','#003153','#CC8899','#E30B5C','#FF0000','#C71585','#FF007F','#E0115F',
    '#92000A','#0F52BA','#C0C0C0','#708090','#A7FC00','#00FF7F','#D2B48C','#483C32','#008080','#40E0D0','#3F00FF','#7F00FF','#40826D','#FFFF00']

    for i in range(k):
        i_data=[data[p] for p in range(n) if partition[p] == i]
        i_data_x=[a[0] for a in i_data]
        i_data_y=[a[1] for a in i_data]
        plt.scatter(i_data_x, i_data_y, s=50, c=colors[i], marker='o', edgecolor='black',label='cluster'+str(i))
    plt.savefig(resPath)
    return

def showPartition(k,data,partition):
    plt.clf()
    n=len(data)
    #colors=['lightgreen','orange','blue','red','lightblue','brown','purple','green','black','white','yellow','indigo','pink','gray','gold']
    colors=['#E52B50','#FFBF00','#9966CC','#FBCEB1','#7FFFD4','#007FFF','#89CFF0','#F5F5DC','#CB4154','#000000','#0095B6','#8A2BE2','#DE5D83','#CD7F32','#964B00','#800020','#702963','#DE3163',
    '#007BA7','#F7E7CE','#7FFF00','#7B3F00','#0047AB','#6F4E37','#FF7F50','#DC143C','#EDC9AF','#50C878','#00FF3F','#FFD700','#808080','#008000','#4B0082','#B57EDC','#FFF700','#C8A2C8','#FF00FF',
    '#FF00AF','#800000','#E0B0FF','#000080','#CC7722','#808000','#FF6600','#FF4500','#DA70D6','#FFE5B4','#CCCCFF','#003153','#CC8899','#E30B5C','#FF0000','#C71585','#FF007F','#E0115F',
    '#92000A','#0F52BA','#C0C0C0','#708090','#A7FC00','#00FF7F','#D2B48C','#483C32','#008080','#40E0D0','#3F00FF','#7F00FF','#40826D','#FFFF00']

    for i in range(k):
        i_data=[data[p] for p in range(n) if partition[p] == i]
        i_data_x=[a[0] for a in i_data]
        i_data_y=[a[1] for a in i_data]
        plt.scatter(i_data_x, i_data_y, s=50, c=colors[i], marker='o', edgecolor='black',label='cluster'+str(i))

    plt.show()
    return

def vizualizeAnchorsOnPartition(k,data,partition,listAnchor,resPath):
    plt.clf()
    n=len(data)
    h=len(listAnchor)
    #colors=['lightgreen','orange','blue','red','lightblue','brown','purple','green','black','white','yellow','indigo','pink','gray','gold']
    colors=['#E52B50','#FFBF00','#9966CC','#FBCEB1','#7FFFD4','#007FFF','#89CFF0','#F5F5DC','#CB4154','#000000','#0095B6','#8A2BE2','#DE5D83','#CD7F32','#964B00','#800020','#702963','#DE3163',
    '#007BA7','#F7E7CE','#7FFF00','#7B3F00','#0047AB','#6F4E37','#FF7F50','#DC143C','#EDC9AF','#50C878','#00FF3F','#FFD700','#808080','#008000','#4B0082','#B57EDC','#FFF700','#C8A2C8','#FF00FF',
    '#FF00AF','#800000','#E0B0FF','#000080','#CC7722','#808000','#FF6600','#FF4500','#DA70D6','#FFE5B4','#CCCCFF','#003153','#CC8899','#E30B5C','#FF0000','#C71585','#FF007F','#E0115F',
    '#92000A','#0F52BA','#C0C0C0','#708090','#A7FC00','#00FF7F','#D2B48C','#483C32','#008080','#40E0D0','#3F00FF','#7F00FF','#40826D','#FFFF00']

    for i in range(k):
        i_data=[data[p] for p in range(n) if partition[p] == i]
        i_data_x=[a[0] for a in i_data]
        i_data_y=[a[1] for a in i_data]

        i_anch=[data[listAnchor[p]] for p in range(h) if partition[listAnchor[p]] == i]
        #print(i_anch)
        i_anch_x=[a[0] for a in i_anch]
        i_anch_y=[a[1] for a in i_anch]

        plt.scatter(i_data_x, i_data_y, s=50, c=colors[i], marker='o', edgecolor='black',label='cluster'+str(i))
        plt.scatter(i_anch_x, i_anch_y, s=170, c=colors[i], marker='^', edgecolor='black',label='cluster'+str(i))
    plt.savefig(resPath)
    return


def vizualizeNeutralAnchors(anchors,k,data,resPath):
    plt.clf()
    #print(anchors)
    h=len(anchors)
    idAnchor=-1
    anchors_x=[a[0] for a in anchors]
    anchors_y=[a[1] for a in anchors]
    n=len(data)

    #colors=['lightgreen','orange','blue','red','lightblue','brown','purple','green','black','white','yellow','indigo','pink','gray','gold']
    colors=['#E52B50','#FFBF00','#9966CC','#FBCEB1','#7FFFD4','#007FFF','#89CFF0','#F5F5DC','#CB4154','#000000','#0095B6','#8A2BE2','#DE5D83','#CD7F32','#964B00','#800020','#702963','#DE3163',
    '#007BA7','#F7E7CE','#7FFF00','#7B3F00','#0047AB','#6F4E37','#FF7F50','#DC143C','#EDC9AF','#50C878','#00FF3F','#FFD700','#808080','#008000','#4B0082','#B57EDC','#FFF700','#C8A2C8','#FF00FF',
    '#FF00AF','#800000','#E0B0FF','#000080','#CC7722','#808000','#FF6600','#FF4500','#DA70D6','#FFE5B4','#CCCCFF','#003153','#CC8899','#E30B5C','#FF0000','#C71585','#FF007F','#E0115F',
    '#92000A','#0F52BA','#C0C0C0','#708090','#A7FC00','#00FF7F','#D2B48C','#483C32','#008080','#40E0D0','#3F00FF','#7F00FF','#40826D','#FFFF00']

    plt.scatter(anchors_x, anchors_y, marker='^', c=colors[0], s=170, edgecolor='black')
    plt.savefig(resPath)

def display(k,p,data):
    colors=['lightgreen','orange','blue','red','lightblue','brown','purple','green','black','white','yellow','indigo','pink','gray','gold']
    for i in range(k):
        plt.scatter(data[p == i][0], data[p == i][1],s=50, c=colors[i],marker='s', edgecolor='black',label='cluster'+str(i))
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

def testAnchors():
    data=[[4,4],[4,5],[5,1],[0,0],[0,1],[1,1],[1,2],[2,2],[3,4]]
    anchors=[[1,1],[2,2],[3,4]]
    h=len(anchors)
    resPath="/home/mathieu/Bureau/testViz.png"
    partition=[2,2,0,0,0,0,1,1,2]
    vizualizeAnchors(anchors,data,partition,resPath)

def vizBoxPlot(order,colorsDic,allResults,path,title,lim):
    plt.close()
    all_data=[]
    labels=[]
    colors=[]
    for i in order:
        if (i in allResults):
            l=allResults[i]
            List_flat = []
            if(type(l[0])==list):
                for a in range(len(l)):
                  for b in range (len(l[a])):
                    List_flat.append(l[a][b])
            else:
                List_flat=l
            print(List_flat)
            all_data.append(np.array(List_flat))
            labels.append(i)
            colors.append(colorsDic[i])
    print(labels)

    fig, axes = plt.subplots(nrows=1, ncols=1)

    # rectangular box plot
    bplot1 = axes.boxplot(all_data,
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=labels,  # will be used to label x-ticks
                             medianprops=dict(color='black') )
    axes.set_title(title)
    #axes.set_aspect(0.5)
    plt.tight_layout()
    if(lim!=None):
        axes.set_ylim([0, lim])

    #colors
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    #save result
    plt.savefig(path+".png")
    plt.show()
    return
