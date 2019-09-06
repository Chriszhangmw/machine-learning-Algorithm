

class HmmCut:
    def __init__(self):
        trans_path = './model/prob_trans.model'
        emit_path = './model/prob_emit.model'
        pi_path = './model/prob_pi.model'
        self.trans = self.load_model(trans_path)
        self.emit = self.load_model(emit_path)
        self.pi = self.load_model(pi_path)

    def load_model(self,model_path):
        f = open(model_path,'r')
        a = f.read()
        word_dict = eval(a)
        f.close()
        return word_dict

    def viterbi(self,x,states,pi,trans,emit):
        dp = [{}]
        path = {}
        for y in states:
            # dp[0][y] = pi[y] * emit[y].get(x[0],0)
            dp[0][y]  = pi[y] * emit[y].get(x[0], 0)
            path[y] = [y]
        for t in range(1,len(x)):
            dp.append({})
            newpath = {}
            for y in states:
                state_path = ([(dp[t-1][y0] * trans[y0].get(y,0) * emit[y].get(x[t],0),y0) for y0 in states if dp[t-1][y0] > 0])
                if state_path == []:
                    (prob,state) = (0.0,'S')
                else:
                    (prob,state) = max(state_path)
                dp[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        (prob,state) = max([(dp[len(x)-1][y],y) for y in states])
        return (prob,path[state])

    # def viterbi(self, obs, states, start_p, trans_p, emit_p):  # 维特比算法（一种递归算法）
    #     # 算法的局限在于训练语料要足够大，需要给每个词一个发射概率,.get(obs[0], 0)的用法是如果dict中不存在这个key,则返回0值
    #     V = [{}]
    #     path = {}
    #     for y in states:
    #         V[0][y] = start_p[y] * emit_p[y].get(obs[0], 0)  # 在位置0，以y状态为末尾的状态序列的最大概率
    #         path[y] = [y]
    #
    #     for t in range(1, len(obs)):
    #         V.append({})
    #         newpath = {}
    #         for y in states:
    #             state_path = ([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emit_p[y].get(obs[t], 0), y0) for y0 in states if
    #                            V[t - 1][y0] > 0])
    #             if state_path == []:
    #                 (prob, state) = (0.0, 'S')
    #             else:
    #                 (prob, state) = max(state_path)
    #             V[t][y] = prob
    #             newpath[y] = path[state] + [y]
    #
    #         path = newpath  # 记录状态序列
    #     (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])  # 在最后一个位置，以y状态为末尾的状态序列的最大概率
    #     return (prob, path[state])  # 返回概率和状态序列

    def cut(self,sent):
        prob,pos_list = self.viterbi(sent,('B','M','E','S'),self.pi,self.trans,self.emit)
        seglist = list()
        word = list()
        for index in range(len(pos_list)):
            if pos_list[index] == 'S':
                word.append(sent[index])
                seglist.append(word)
                word = []
            elif pos_list[index] in ['B','M']:
                word.append(sent[index])
            elif pos_list[index] == 'E':
                word.append(sent[index])
                seglist.append(word)
                word = []
        seglist = [''.join(word) for word in seglist]
        return seglist

if __name__ == '__main__':
    sent = '目前在自然语言处理技术中，中文处理技术比西文处理技术要落后很大一段距离，许多西文的处理方法中文不能直接采用'
    cuter = HmmCut()
    seglist = cuter.cut(sent)
    print(seglist)












