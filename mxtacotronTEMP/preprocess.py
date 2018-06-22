from sklearn import preprocessing
import string
import numpy as np
class PreProcessText:
    def __init__(self, validCharType):
        self.charType = validCharType
        self.labelencoder = preprocessing.LabelEncoder()
        self.validChars = ['*', ' ',',','.','!','?']
        if validCharType == 'all':
            self.validChars.append('@')
            self.validChars.append(')')
            self.validChars.append('#')
            self.validChars.append('(')
            for i in range(26):
                self.validChars.append(string.ascii_lowercase[i])
                self.validChars.append(string.ascii_uppercase[i])
        elif validCharType == 'allcase':
            for i in range(26):
                self.validChars.append(string.ascii_lowercase[i])
                self.validChars.append(string.ascii_uppercase[i])
        elif validCharType == 'lowercaseonly':
            for i in range(26):
                self.validChars.append(string.ascii_lowercase[i])
        
        self.labelencoder.fit(self.validChars)

    
    def Encode(self, s):
        
        def SubsUnknown(s, le):
            if s not in le.classes_:
                return '*'
            else:
                return s
        
        stringToEncode = s

        if self.charType == 'lowercaseonly':
            stringToEncode = stringToEncode.lower()
            
        stringToEncode = list(stringToEncode)
        stringToEncode = [SubsUnknown(item, self.labelencoder) for item in stringToEncode]
        encodedText = np.array(self.labelencoder.transform(stringToEncode)).astype(int)
        return encodedText