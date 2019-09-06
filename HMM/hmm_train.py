

class HmmModel:
    def __init__(self):
        self.line_index = -1
        self.char_set = set()

    def init(self):
        trans_dict = {}
        emit_dict = {}
        count_dict = {}
        pi_dict = {}
        state_list = ['B','M','E','S']

        for state in state_list:
            trans_dict[state] = {}
            for state2 in state_list:
                trans_dict[state][state2] = 0.0
        for state in state_list:
            emit_dict[state] = {}
            pi_dict[state] = 0.0
            count_dict[state] = 0

        return trans_dict,emit_dict,pi_dict,count_dict

    def save_model(self,word_dict,model_path):
        f = open(model_path,'w')
        f.write(str(word_dict))
        f.close()

    def get_word_status(self,word):
        word_status = []
        if len(word) == 1:
            word_status.append('S')
        elif len(word) == 2:
            word_status = ['B','E']
        else:
            number_m = len(word)-2
            m_list = ['M'] * number_m
            word_status.append('S')
            word_status.extend(m_list)
            word_status.append('E')
        return word_status

    def train(self,train_filepath,trans_path,emit_path,pi_path):
        trans_dict, emit_dict, pi_dict, count_dict = self.init()
        n = 0
        for line in open(train_filepath,encoding='utf-8'):
            self.line_index +=1
            n +=1
            line = line.strip()
            if not line:
                continue
            char_list = []
            for i in range(len(line)):
                if line[i] == ' ':
                    continue
                char_list.append(line[i])

            self.char_set = set(char_list)#也就是所有的X
            word_list = line.split(' ')
            line_status = []
            for word in word_list:
                line_status.extend(self.get_word_status(word))
            if len(char_list) == len(line_status):
                for i in range(len(line_status)):
                    if i == 0:
                        pi_dict[line_status[0]] +=1
                        count_dict[line_status[0]] +=1
                    else:
                        trans_dict[line_status[i-1]][line_status[i]] +=1
                        count_dict[line_status[i]] +=1
                        if char_list[i] not in emit_dict[line_status[i]]:
                            emit_dict[line_status[i]][char_list[i]] = 0.0
                        else:
                            emit_dict[line_status[i]][char_list[i]] +=1
                else:
                    continue
        print(n)

        for key in pi_dict:
            print(self.line_index)
            pi_dict[key] = pi_dict[key] * 1.0 /self.line_index
        for key in trans_dict:
            for key2 in trans_dict[key]:
                trans_dict[key][key2] = trans_dict[key][key2] * 1.0 / count_dict[key]
        for key in emit_dict:
            for word in emit_dict[key]:
                emit_dict[key][word] =  emit_dict[key][word] * 1.0 / count_dict[key]
        self.save_model(trans_dict,trans_path)
        self.save_model(emit_dict,emit_path)
        self.save_model(pi_dict,pi_path)

        return trans_dict,emit_dict,pi_dict


if __name__ == '__main__':
    train_path = './data/train.txt'
    trans_path = './model/prob_trans.model'
    emit_path = './model/prob_emit.model'
    pi_path = './model/prob_pi.model'
    trainer = HmmModel()

    trainer.train(train_path,trans_path,emit_path,pi_path)
















