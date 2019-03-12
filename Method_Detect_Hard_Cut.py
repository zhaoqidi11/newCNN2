class RAIDataset_GenerateTrainingDataset():

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
        BatchSize = 200
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


            # if i*GroupLength>=14100:
            #     print "a"
            MIUL = np.mean(d[GroupLength*i:GroupLength*i+GroupLength])
            SigmaL = np.std(d[GroupLength*i:GroupLength*i+GroupLength])

            Tl.append(MIUL + a * ( 1 + math.log( MIUG / MIUL ) ) * SigmaL)
            # Tl.append(1.1 * MIUL + 0.6 * (MIUG/MIUL) * SigmaL)
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

    def EuclideanDistanceBetweenTwoFramesThroughCNN(self, frame1, frame2):
        import sys
        import numpy as np
        sys.path.insert(0, 'C:\\newcaffe\\caffe\\python')
        import caffe

        HybridCNN_Siamese_Def = '/media/user02/New Volume/Meisa/squeezenet/deploy.prototxt'
        HybridCNN_Siamese_Weight = '/media/user02/New Volume/Meisa/squeezenet/squeezenet_v1.1.caffemodel'

        WindowsPath = 'E:\\Meisa_SiameseNetwork\\CNN2\\'

        HybridCNN_Siamese_Def = WindowsPath + 'new_Siamese_hybridCNN_deploy_new.prototxt'
        HybridCNN_Siamese_Weight = WindowsPath + 'snapshot_iter_10000.caffemodel'

        net = caffe.Net(HybridCNN_Siamese_Def,
                        HybridCNN_Siamese_Weight,
                        caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data_left'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        # mu = np.array([104, 117, 123]) # Mean Value from "ImageNet2012"
        # transformer.set_mean('data', mu)
        # transformer.set_raw_scale('data', 255)
        # transformer.set_channel_swap('data', (2, 1, 0))

        # It save the batch size
        BatchSize = 1
        net.blobs['data_left'].reshape(BatchSize,
                                  3,
                                  227, 227)
        net.blobs['data_right'].reshape(BatchSize,
                                  3,
                                  227, 227)

        data_left = transformer.preprocess('data', frame1)
        data_right = transformer.preprocess('data', frame2)

        net.blobs['data_left'].data[...] = data_left
        net.blobs['data_right'].data[...] = data_right

        output = net.forward()

        return output['EuclideanDistance']


    def CTDetectionBaseOnHist(self, VideoPath, HardCutTruth, GradualTruth):
        import numpy as np
        import cv2
        import sys
        caffe_root = '/media/user02/New Volume/caffe'
        sys.path.insert(0, caffe_root + '/python')
        import caffe

        caffe.set_mode_gpu()
        caffe.set_device(0)

        k = 0.4
        Tc = 0.05

        # CandidateSegments = self.CutVideoIntoSegments()

        CandidateSegments = self.CutVideoIntoSegmentsBaseOnNeuralNet(VideoPath)

        # self.CheckSegments(CandidateSegments, HardCutTruth, GradualTruth)

        # It saves the predicted shot boundaries
        Answer = []

        # It saves the candidate segments which may have gradual
        CandidateGra = []

        i_Video = cv2.VideoCapture(VideoPath)

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        # get the number of frames of this video
        FrameNum = int(i_Video.get(7))

        # It saves the predicted transition numbers
        AnswerLength = 0

        WindowsPath = 'E:\\Meisa_SiameseNetwork\\CNN2\\'

        HybridCNN_Siamese_Def = WindowsPath + 'Example\\'+'deploy_bn_new_Siamese_hybridCNN_deploy_new.prototxt'
        HybridCNN_Siamese_Weight = WindowsPath + 'snapshot_iter_10000.caffemodel'

        HybridCNN_Siamese_Def = '/media/user02/New Volume/deploy_bn_new_Siamese_hybridCNN_deploy_new.prototxt'
        HybridCNN_Siamese_Weight = '/media/user02/New Volume/SnapShot2/snapshot_iter_10000.caffemodel'


        net = caffe.Net(HybridCNN_Siamese_Def,
                        HybridCNN_Siamese_Weight,
                        caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data_left'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))


        CandidateFrames1 = []
        CandidateFrames2 = []
        for i in range(len(CandidateSegments)):
            frame1add = 0
            frame2add = 0
            # frame1 saves the first frame of the segment's
            i_Video.set(1, CandidateSegments[i][0])
            ret1, frame1 = i_Video.read()

            # Consider the situation that the frame that would be not extracted
            while frame1 is None:
                frame1add += 1
                i_Video.set(1, CandidateSegments[i][0] + frame1add)
                ret1, frame1 = i_Video.read()

            # frame2 saves the last frame of the segment's
            i_Video.set(1, CandidateSegments[i][1])
            ret1, frame2 = i_Video.read()

            # Consider the situation that the frame that would be not extracted
            while frame2 is None:
                frame2add += 1
                i_Video.set(1, CandidateSegments[i][1] - frame2add)
                ret1, frame2 = i_Video.read()

            HistDifference = []

            # if CandidateSegments[i][0]>=14130:
            # print 'a'
            if self.getHist_Manhattan(frame1, frame2, wid * hei) >= 0.45:
                # Calculate the Manhattan distance from the frame1 and frame2 (Hist)
                for j in range(CandidateSegments[i][0], CandidateSegments[i][1]):

                    i_Video.set(1, j)
                    ret1_, frame1_ = i_Video.read()

                    i_Video.set(1, j + 1)
                    ret2_, frame2_ = i_Video.read()

                    HistDifference.append(self.getHist_chi_square(frame1_, frame2_, wid * hei))

                if np.max(HistDifference) > 0.1:  # and len([_ for _ in HistDifference if _>0.1])<len(HistDifference):
                    CandidatePeak = -1
                    MAXValue = -1

                    # Spectial Situation #1
                    if HistDifference[0] > 0.1 and HistDifference[0] > HistDifference[1]:
                        CandidatePeak = 0
                        MAXValue = HistDifference[0] - HistDifference[1]

                    for ii in range(1, len(HistDifference) - 1):
                        if HistDifference[ii] > 0.1 and HistDifference[ii] > HistDifference[ii - 1] and HistDifference[
                            ii] > HistDifference[ii + 1]:
                            if np.max([np.abs(HistDifference[ii] - HistDifference[ii - 1]),
                                       np.abs(HistDifference[ii] - HistDifference[ii + 1])]) > MAXValue:
                                CandidatePeak = ii
                                MAXValue = np.max([np.abs(HistDifference[ii] - HistDifference[ii - 1]),
                                                   np.abs(HistDifference[ii] - HistDifference[ii + 1])])

                    if HistDifference[-1] > 0.1 and HistDifference[-1] > HistDifference[-2] and (
                            HistDifference[-1] - HistDifference[-2]) > MAXValue:
                        CandidatePeak = len(HistDifference) - 1
                        MAXValue = HistDifference[-1] - HistDifference[-2]

                    if MAXValue > -1:
                        i_Video.set(1, CandidateSegments[i][0] + CandidatePeak)
                        CandidateFrame1_ret1, CandidateFrame1 = i_Video.read()

                        i_Video.set(1,  CandidateSegments[i][0] + CandidatePeak + 1)
                        CandidateFrame1_ret2, CandidateFrame2 = i_Video.read()

                        if len(Answer) == 0:
                            CandidateFrames1 = np.array([transformer.preprocess('data', CandidateFrame1)])
                            CandidateFrames2 = np.array([transformer.preprocess('data', CandidateFrame2)])
                        else:

                            CandidateFrames1= np.concatenate(
                                [CandidateFrames1, np.array([transformer.preprocess('data', CandidateFrame1)])])
                            CandidateFrames2 = np.concatenate(
                                [CandidateFrames2, np.array([transformer.preprocess('data', CandidateFrame2)])])


                        Answer.append(
                            ([CandidateSegments[i][0] + CandidatePeak, CandidateSegments[i][0] + CandidatePeak + 1]))
        net.blobs['data_left'].reshape(len(Answer),3,227,227)
        net.blobs['data_right'].reshape(len(Answer),3,227,227)

        net.blobs['data_left'].data[...] = CandidateFrames1
        net.blobs['data_right'].data[...] = CandidateFrames2

        output = net.forward()
        New_Answer = []
        for i in range(len(Answer)):
            if output['EuclideanDistance'][i]>0.001:
                New_Answer.append(Answer[i])


        [cut_correct, gradual_correct, all_correct] =self.eval(New_Answer, HardCutTruth)

        return len(HardCutTruth)-cut_correct

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

    def DetectionOnRAIDataset(self):
        from glob import glob

        LabelFilePath = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\gt_'
        VideoPath = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\videos\\'

        LabelFilePath = '/media/user02/New Volume/RAIDataset/gt_'
        VideoPath = '/media/user02/New Volume/RAIDataset/'

        AllHard = 0
        AllMiss = 0

        VideosNumber = 10

        for i in range(1, VideosNumber+1):
            [HardTruth, GraTruth] = self.GetLabels(LabelFilePath + str(i) + '.txt')

            Miss = self.CTDetectionBaseOnHist(VideoPath + str(i) +'.mp4', HardTruth, GraTruth)
            AllHard += len(HardTruth)
            AllMiss += Miss


if __name__ == '__main__':
    test1 = RAIDataset_GenerateTrainingDataset()
    # test1.CTDetection()
    # test1.CutVideoIntoSegments()

    # test1.CTDetectionBaseOnHist()
    test1.DetectionOnRAIDataset()
    # test1.CheckSegments(test1.CutVideoIntoSegmentsBaseOnNeuralNet())