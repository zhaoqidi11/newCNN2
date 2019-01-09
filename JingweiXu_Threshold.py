class JingweiXu():


    def get_vector(self, segments):
        import sys
        import os
        sys.path.insert(0, '/data/caffe/python')
        import caffe
        import cv2


        caffe.set_mode_gpu()
        caffe.set_device(0)
        # load model(.prototxt) and weight (.caffemodel)

        # os.chdir('/data/Meisa/ResNet/ResNet-50')
        # ResNet_Weight = './resnet50_cvgj_iter_320000.caffemodel'  # pretrained on il 2012 and place 205

        os.chdir('/data/Meisa/hybridCNN')
        Hybrid_Weight = './hybridCNN_iter_700000.caffemodel'



        # ResNet_Def = 'deploynew_globalpool.prototxt'

        Hybrid_Def = 'Shot_hybridCNN_deploy_new.prototxt'

        Alexnet_Def = '/data/alexnet/deploy_alexnet_places365.prototxt.txt'
        Alexnet_Weight = '/data/alexnet/alexnet_places365.caffemodel'
        net = caffe.Net(Hybrid_Def,
                        Hybrid_Weight,
                        caffe.TEST)

        # load video
        i_Video = cv2.VideoCapture(self.Video_path)

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        # get the number of frames of this video
        framenum = int(i_Video.get(7))

        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')




        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        net.blobs['data'].reshape(1,
                                  3,
                                  227, 227)

        FrameV = []

        if len(segments) == 1:
            i_Video.set(1, segments[0])
            ret, frame = i_Video.read()
            if frame is None:
                print i
            transformed_image = transformer.preprocess('data', frame)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            FrameV.extend(output['fc8'][0].tolist())
            #FrameV.extend(np.squeeze(output['global_pool'][0]).tolist())
            return FrameV

        for i in range(segments[0], segments[1]+1):
            i_Video.set(1, i)
            ret, frame = i_Video.read()
            if frame is None:
                print i
                continue
            transformed_image = transformer.preprocess('data', frame)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            FrameV.append(output['fc8'][0].tolist())
            # FrameV.append(np.squeeze(output['global_pool'][0]).tolist())

        return FrameV


    def RGBToGray(self, RGBImage):

        import numpy as np
        return np.dot(RGBImage[..., :3], [0.299, 0.587, 0.114])


    # Get the Manhattan Distance
    def Manhattan(self, vector1, vector2):
        import numpy as np
        return np.sum(np.abs(vector1 - vector2))


    # Get the Color Histogram from "frame"
    def GetFrameHist(self, frame, binsnumber):
        import cv2
        Bframehist = cv2.calcHist([frame], channels=[0], mask=None, ranges=[0.0, 255.0], histSize=[binsnumber])
        Gframehist = cv2.calcHist([frame], channels=[1], mask=None, ranges=[0.0, 255.0], histSize=[binsnumber])
        Rframehist = cv2.calcHist([frame], channels=[2], mask=None, ranges=[0.0, 255.0], histSize=[binsnumber])
        return [Bframehist, Gframehist, Rframehist]

    # Get the Manhattan distance between the histogram of frame1 and frame2
    def getHist_Manhattan(self, frame1, frame2, allpixels):

        binsnumber = 64

        [Bframe1hist, Gframe1hist, Rframe1hist] = self.GetFrameHist(frame1, binsnumber)
        [Bframe2hist, Gframe2hist, Rframe2hist] = self.GetFrameHist(frame2, binsnumber)

        distance_Manhattan = self.Manhattan(Bframe1hist, Bframe2hist) + self.Manhattan(Gframe1hist,
                                                                                       Gframe2hist) + self.Manhattan(
            Rframe1hist, Rframe2hist)
        return distance_Manhattan / allpixels

    # Get the chi square distance between the histogram of frame1 and frame2
    def getHist_chi_square(self, frame1, frame2, allpixels):
        import cv2
        binsnumber = 64

        [Bframe1hist, Gframe1hist, Rframe1hist] = self.GetFrameHist(frame1, binsnumber)
        [Bframe2hist, Gframe2hist, Rframe2hist] = self.GetFrameHist(frame2, binsnumber)

        chi_square_distance = cv2.compareHist(Bframe1hist, Bframe2hist, method=cv2.HISTCMP_CHISQR) + cv2.compareHist(
            Gframe1hist, Gframe2hist, method=cv2.HISTCMP_CHISQR) + cv2.compareHist(Rframe1hist, Rframe2hist,
                                                                                   method=cv2.HISTCMP_CHISQR)
        return chi_square_distance / (allpixels)


    def CutVideoIntoSegments(self):
        import math
        import cv2
        import numpy as np

        # It save the pixel intensity between 20n and 20(n+1)
        d = []
        SegmentsLength = 11
        i_Video = cv2.VideoCapture(self.Video_path)

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')

        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        # The number of segments
        Count = int(math.ceil(float(FrameNumber) / float(SegmentsLength-1)))
        for i in range(Count):

            i_Video.set(1, (SegmentsLength-1)*i)
            ret1, frame_20i = i_Video.read()

            if((SegmentsLength-1)*(i+1)) >= FrameNumber:
                i_Video.set(1, FrameNumber-1)
                ret2, frame_20i1 = i_Video.read()
                # d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))

                d.append(self.getHist(frame_20i, frame_20i1, wid*hei))
                break

            i_Video.set(1, (SegmentsLength-1)*(i+1))
            ret2, frame_20i1 = i_Video.read()

            # d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))
            d.append(self.getHist(frame_20i, frame_20i1, wid*hei))


        GroupLength = 10

        # The number of group
        GroupNumber = int(math.ceil(float(len(d)) / float(GroupLength)))

        MIUG = np.mean(d)
        a = 0.5 # The range of a is 0.5~0.7
        Tl = [] # It save the Tl of each group
        CandidateSegment = []
        for i in range(GroupNumber):



            MIUL = np.mean(d[GroupLength*i:GroupLength*i+GroupLength])
            SigmaL = np.std(d[GroupLength*i:GroupLength*i+GroupLength])

            Tl.append(MIUL + a*(1+math.log(MIUG/MIUL))*SigmaL)
            for j in range(GroupLength):
                if i*GroupLength + j >= len(d):
                    break
                if d[i*GroupLength+j]>Tl[i]:
                    CandidateSegment.append([(i*10+j)*(SegmentsLength-1), (i*10+j+1)*(SegmentsLength-1)])
                    #print 'A candidate segment is', (i*10+j)*20, '~', (i*10+j+1)*20


        for i in range(1,len(d)-1):
            if (d[i]>(3*d[i-1]) or d[i]>(3*d[i+1])) and d[i]> 0.8 * MIUG:
                if [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)] not in CandidateSegment:
                    j = 0
                    while j < len(CandidateSegment):
                        if (i+1)*(SegmentsLength-1)<= CandidateSegment[j][0]:
                            CandidateSegment.insert(j, [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)])
                            break
                        j += 1
        return CandidateSegment
        #print 'a'

    def CutVideoIntoSegmentsBaseOnNeuralNet(self, Video_path):
        import math
        import cv2
        import numpy as np
        import copy

        import sys
        sys.path.insert(0, '/media/user02/New Volume/caffe/python')

        import caffe

        caffe.set_mode_gpu()
        caffe.set_device(0)

        # caffe.set_mode_cpu()

        SqueezeNet_Def = '/media/user02/New Volume/Meisa/squeezenet/deploy.prototxt'
        SqueezeNet_Weight = '/media/user02/New Volume/Meisa/squeezenet/squeezenet_v1.1.caffemodel'

        # WindowsPath = 'E:\\Meisa_SiameseNetwork\\SqueezeNet\\'
        #
        # SqueezeNet_Def = WindowsPath + 'deploy.prototxt'
        # SqueezeNet_Weight = WindowsPath + 'squeezenet_v1.1.caffemodel'

        net = caffe.Net(SqueezeNet_Def,
                        SqueezeNet_Weight,
                        caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        mu = np.array([104, 117, 123]) # Mean Value from "ImageNet2012"
        transformer.set_mean('data', mu)
        # transformer.set_raw_scale('data', 255)
        # transformer.set_channel_swap('data', (2, 1, 0))

        # It save the batch size
        BatchSize = 100
        net.blobs['data'].reshape(BatchSize,
                                  3,
                                  227, 227)

        # It save the pixel intensity between 20n and 20(n+1)
        d = []

        SegmentsLength = 11

        i_Video = cv2.VideoCapture(Video_path)
        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')

        # get width of this video
        wid = int(i_Video.get(3))
        # get height of this video
        hei = int(i_Video.get(4))
        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        # The number of segments
        Count = int(math.ceil(float(FrameNumber) / float(SegmentsLength-1)))

        FrameSqueezeNetOUT = []

        if Count >= BatchSize:
            for i in range(Count - Count % BatchSize):
                if i % BatchSize == 0:
                    i_Video.set(1, (SegmentsLength - 1) * i)
                    ret0, frame0 = i_Video.read()
                    Frame_Eigenvector = np.array([transformer.preprocess('data', frame0)])
                else:
                    i_Video.set(1, (SegmentsLength - 1) * i)
                    ret, frame = i_Video.read()
                    Frame_Eigenvector = np.concatenate([Frame_Eigenvector, np.array([transformer.preprocess('data',frame)])])
                if i % BatchSize == BatchSize-1:
                    net.blobs['data'].data[...] = Frame_Eigenvector
                    output = net.forward()

                    FrameSqueezeNetOUT.extend(np.squeeze(copy.deepcopy(output['pool10'])))

        NewCount = Count % BatchSize
        if NewCount > 0:

            i_Video.set(1, Count - Count % BatchSize)
            ret_new0, frame_new0 = i_Video.read()

            Frame_Eigenvector = np.array([transformer.preprocess('data', frame_new0)])
            for i in range(Count - Count % BatchSize+1, Count):

                i_Video.set(1, Count - Count % BatchSize)
                ret_new, frame_new = i_Video.read()

                Frame_Eigenvector = np.concatenate([Frame_Eigenvector, np.array([transformer.preprocess('data', frame_new)])])

            net.blobs['data'].reshape(NewCount,
                                      3,
                                      227, 227)
            net.blobs['data'].data[...] = Frame_Eigenvector

            output = net.forward()

            if output['pool10'].shape[0] == 1:
                FrameSqueezeNetOUT.append(copy.deepcopy(np.squeeze(output['pool10']).reshape(1000)))
            else:
                FrameSqueezeNetOUT.extend(copy.deepcopy(np.squeeze(output['pool10'])))

        for i in range(Count-1):
            d.append(self.cosin_distance(FrameSqueezeNetOUT[i].tolist(), FrameSqueezeNetOUT[i+1].tolist()))

        GroupLength = 20
        # The number of group
        GroupNumber = int(math.ceil(float(len(d)) / GroupLength))

        MIUG = np.mean(d)
        a = 0.7 # The range of a is 0.5~0.7
        Tl = [] # It save the Tl of each group
        CandidateSegment = []
        for i in range(GroupNumber):


            if i*GroupLength>=14100:
                print "a"
            MIUL = np.mean(d[GroupLength*i:GroupLength*i+GroupLength])
            SigmaL = np.std(d[GroupLength*i:GroupLength*i+GroupLength])

            Tl.append(MIUL + a * ( 1 + math.log( MIUG / MIUL ) ) * SigmaL)
            # Tl.append(1.1 * MIUL + 0.6 * (MIUG / MIUL) * SigmaL)
            for j in range(GroupLength):
                if i*GroupLength + j >= len(d):
                    break
                if d[i*GroupLength+j]<Tl[i]:
                    CandidateSegment.append([(i*GroupLength+j)*(SegmentsLength-1), (i*GroupLength+j+1)*(SegmentsLength-1)])



        for i in range(1,len(d)-1):
            if (d[i]>(3*d[i-1]) or d[i]>(3*d[i+1])) and d[i]> 0.8 * MIUG:
                if [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)] not in CandidateSegment:
                    j = 0
                    while j < len(CandidateSegment):
                        if (i+1)*(SegmentsLength-1)<= CandidateSegment[j][0]:
                            CandidateSegment.insert(j, [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)])
                            break
                        j += 1

        return CandidateSegment
        #print 'a'

    # Calculate the cosin distance between vector1 and vector2
    def cosin_distance(self, vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)

    # Calculate the D1
    def getD1(self, Segment):
        return self.cosin_distance(Segment[0], Segment[-1])


####################################The Following is used for evaluating################################################
    def if_overlap(self, begin1, end1, begin2, end2):
        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 >= begin2


    def get_union_cnt(self,set1, set2):
        cnt = 0
        for begin, end in set1:
            for _begin, _end in set2:
                if self.if_overlap(begin, end, _begin, _end):
                    cnt += 1
                    break
        return cnt

    def recall_pre_f1(self,a, b, c):
        a = float(a)
        b = float(b)
        c = float(c)
        recall = a / b if b != 0 else 0
        precison = a / c if c != 0 else 0
        f1 = 2 * recall * precison / (recall + precison)
        return precison, recall, f1

    def eval(self, predict, gt):


        gt_cuts = [(begin,end) for begin,end in gt if end-begin==1]
        gt_graduals = [(begin, end) for begin, end in gt if end - begin > 1]

        predicts_cut = [(begin,end) for begin,end in predict if end-begin==1]
        predicts_gradual = [(begin, end) for begin, end in predict if end - begin > 1]

        cut_correct = self.get_union_cnt(gt_cuts, predicts_cut)
        gradual_correct = self.get_union_cnt(gt_graduals, predicts_gradual)
        all_correct = self.get_union_cnt(predicts_cut + predicts_gradual, gt)

        return [cut_correct, gradual_correct, all_correct]

    ##################################################################################



    # Check the segments selected (by the function called CutVideoIntoSegments) whether have cut
    def CheckSegments(self, CandidateSegments, HardCutTruth, GradualTruth):

        import numpy as np
        MissHard = []
        MissGra = []
        for i in range(len(HardCutTruth)):
            for j in range(len(CandidateSegments)):
                if CandidateSegments[j][1] < HardCutTruth[i][0]:
                    continue
                if self.if_overlap(CandidateSegments[j][0], CandidateSegments[j][1], HardCutTruth[i][0],
                                   HardCutTruth[i][1]):
                    break
                if CandidateSegments[j][0] > HardCutTruth[i][1]:
                    MissHard.append(HardCutTruth[i])
                    break

        for i in range(len(GradualTruth)):
            for j in range(len(CandidateSegments)):
                if CandidateSegments[j][1] < GradualTruth[i][0]:
                    continue
                if self.if_overlap(CandidateSegments[j][0], CandidateSegments[j][1], GradualTruth[i][0],
                                   GradualTruth[i][1]):
                    break
                if CandidateSegments[j][0] > GradualTruth[i][1]:
                    MissGra.append(GradualTruth[i])
                    break

        print "MissHard No. is ", len(MissHard)
        print "MissGra No. is ", len(MissGra)
        if len(HardCutTruth)>0:
            print 'Hard Rate is ', (len(HardCutTruth) - len(MissHard)) / float(len(HardCutTruth))
        if len(GradualTruth)>0:
            print 'Gra Rate is ', (len(GradualTruth) - len(MissGra)) / float(len(GradualTruth))

        return [MissHard, MissGra]



    # CT Detection base on CNN
    def CTDetection(self):
        import math
        import matplotlib.pyplot as plt
        import numpy as np

        k = 0.4
        Tc = 0.05

        CandidateSegments = self.CutVideoIntoSegments()
        # for i in range(len(CandidateSegments)):
        #     FrameV = self.get_vector(CandidateSegments[i])
        [HardCutTruth, GradualTruth] = self.CheckSegments(CandidateSegments)

        # It save the predicted shot boundaries
        Answer = []

        # It save the candidate segments which may have gradual
        CandidateGra = []

        for i in range(len(CandidateSegments)):
            FrameV = []
            FrameV.append(self.get_vector([CandidateSegments[i][0]]))
            FrameV.append(self.get_vector([CandidateSegments[i][-1]]))

            D1 = self.getD1(FrameV)
            if D1 < 0.9:
                D1Sequence = []

                CandidateFrame = self.get_vector(CandidateSegments[i])
                for j in range(len(CandidateFrame) - 1):
                    D1Sequence.append(self.cosin_distance(CandidateFrame[j], CandidateFrame[j+1]))

                if len([_ for _ in D1Sequence if _ < 0.9]) > 1:
                    CandidateGra.append([CandidateSegments[i][0],CandidateSegments[i][0]+20])
                    continue
                if np.min(D1Sequence) < k*D1+(1-k):
                    if np.max(D1Sequence) - np.min(D1Sequence) >  Tc:
                        Answer.append([CandidateSegments[i][0]+np.argmin(D1Sequence), CandidateSegments[i][0]+np.argmin(D1Sequence)+1])
                    else:
                        CandidateGra.append([CandidateSegments[i][0], CandidateSegments[i][0] + 20])
                else:
                    CandidateGra.append([CandidateSegments[i][0], CandidateSegments[i][0] + 20])

                    #if np.max(D1Sequence)- np.min(D1Sequence) > Tc:
                        #print np.argmin(D1Sequence)


        Miss = 0
        True = 0
        False = 0
        for i in Answer:
            if i not in HardCutTruth:
                print 'False :', i, '\n'
                False = False + 1
            else:
                True = True + 1

        for i in HardCutTruth:
            if i not in Answer:
                Miss = Miss + 1

        print 'False No. is', False,'\n'
        print 'True No. is', True, '\n'
        print 'Miss No. is', Miss, '\n'

        [cut_correct, gradual_correct, all_correct] =self.eval(Answer, HardCutTruth)
        print self.recall_pre_f1(cut_correct, len(HardCutTruth), len(Answer))
            # # plot the image
            #
            # x = range(len(D1Sequence))
            #
            # plt.figure()
            # plt.plot(x, D1Sequence)
            #
            # plt.show()
    def GetLabels(self, LabelTXT):
        HardTruth = []
        GraTruth = []
        with open(LabelTXT) as f:
            AllLines = f.readlines()
        GroundTruth = [[int(AllLines[0].strip().split('\t')[-1])]]
        for i in range(1, len(AllLines)-1):
            GroundTruth[-1].extend([int(AllLines[i].strip().split('\t')[0])])
            GroundTruth.append([int(AllLines[i].strip().split('\t')[1])])
        GroundTruth[-1].extend([int(AllLines[-1].strip().split('\t')[0])])

        for i in GroundTruth:
            if i[1]-i[0] == 1:
                HardTruth.append(i)
            else:
                GraTruth.append(i)

        return [HardTruth, GraTruth]

    def DetectionOnRAIData(self):
        RAIDatasetPath = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\videos\\'
        LabelFilePath = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\'

        RAIDatasetPath = '/media/user02/New Volume/Meisa/ShotDetector/video_rai/'
        LabelFilePath = '/media/user02/New Volume/Meisa/ShotDetector/video_rai/'
        from glob import glob

        AllLabels = glob(LabelFilePath + 'gt_*.txt')
        AllVideos = glob(RAIDatasetPath + '*.mp4')

        AllHard = 0
        AllGra = 0
        AllMissHard = 0
        AllMissGra = 0
        AllCandidateSegments = 0
        for i in range(len(AllLabels)):
            [HardTruth, GraTruth] = self.GetLabels(AllLabels[i])

            AllHard += len(HardTruth)
            AllGra += len(GraTruth)

            CandidateSegments = self.CutVideoIntoSegmentsBaseOnNeuralNet(AllVideos[i])
            [MissHard, MissGra] = self.CheckSegments(CandidateSegments, HardTruth, GraTruth)

            AllMissHard += len(MissHard)
            AllMissGra += len(MissGra)
            AllCandidateSegments += len(CandidateSegments)

            print 'Now the No. of Candidate Segments is', AllCandidateSegments
            print 'Now the recall of hard is', (AllHard - AllMissHard) / float(AllHard)
            print 'Now the recall of gra is', (AllGra - AllMissGra) / float(AllGra)
if __name__ == '__main__':
    test1 = JingweiXu()
    # test1.CTDetection()
    # test1.CutVideoIntoSegments()

    # test1.CTDetectionBaseOnHist()
    test1.DetectionOnRAIData()
    # test1.CheckSegments(test1.CutVideoIntoSegmentsBaseOnNeuralNet())