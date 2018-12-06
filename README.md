# keras-lstm-text-generator-website
A LSTM neural network, trained on songbook corpses, now generates song lyrics in author`s style.

Neural network evaluates straight in your browser by keras.js library.
It is strongly recommended to run on a machine with strong GPU.

Technologies used:
- Tensorflow
- Keras
- Keras.js
- JQuery
- Bootstrap

Languages used:
- Python
- PHP
- Javascript
- HTML/CSS

Put all the /www folder to any webserver with PHP to check how it works.
Folder /lstm contains python code to train your word or character level LSTM networks.
Convert .h5 files to .bin models using keras.js converter.

Examples of generated texts:

Scorpions
```
and the time my life will take you
come, to take some night 
as the pole is gone
we wants to be suxmised 
don't give and all the way to my love
is there anybody there who feels that vibration
who with me here and in the evening when i come back to love ya

an just love to me right now
love me till i'm down
and i wonder if you know it too 
i skywrite your name way
not the way it find your way
and you look us in cotes
of the bind is time
no one like you
i can't wait for the nights with you
i imagine the things that we've been take
you'll be the rover flows
```

Large ru/en song lyrics corpse
```
and break your pantary daystone
i believe one exact my life
may be a light, and turn to the home,
square hide out of sight
the wind is the money
that i should be quite so come out with the steps off my rain
from the house when you make it all new joe en handoners
i recogniles of the night.
this is the army
then he knew too much
strikin' it up, oh god keep on seat
```

Michael Jackson
```
(dead on like another day)
you're playin' workin' disappear
sweet the beat where i cannot get it!
take me, tell of change
they won't go look around
but now when they've been waiting on- hey baby
she wants to run every way
hey i'm undering and i'm not dreamin'
your very say

our love will said you always come and get yourself behind
oh no, gursin my man
i cannot come home at it
too much for me!

i'll plant you a game of life without love
slime's on the man slipping
the birds are the sun
in the heart: on ang bursed myself long as i can
```

DDT
```
боже, как хорошо, как легки эти двери, я круги держа и слова.
не дожить, не допеть, не дает этот город разбился вчера, время выключить к тебе мой насос
шелеском в окно.
как на огонь вода!
сплющила рожу оконным стеклом в летаргическом сне, я часто не верю в пожатие рук.
и, видя жизнь восколько мне без тебя - сгорим, утонем, сгинет урожай
ждать не сможешь - проиграем мы войну, эх, гражданка по большой "гражданке", как чёрное перо.
эй пляши, пляши, старуха, недовня дрянь небо свободы обернись и ты прохожий,
тает снег, дороги-годы, антонина, мы похожи.
```

Common Russian Rock corpse
```
у нас есть приятно в углу с окном,
и в голове гитара...
и кончится камень с позабытым сытом;
я встречаю рассвет,
каждый день,
пускай в неисколого звонка и ночь зла для тебя
где же ты пройдешь тебе свою роль,
чтоб высохоты околомчится мы спать на кровать.

звёзда живёт в сеть остаться одного.
у меня до смерти
он идет к своей волчице.
поздно, может быть в первый раз крикнет конца.
только дым,
и ставить вином,
что стоит за тобой.
и в алтарь.
ты сказала, ты снимаешь меня: всегда моя!
в ответ "нет, не ты, его помнют черной звездой,
а потом было у нас,
уплывай, ее кончается мозг

а давность причина всех!
на бездумно помнит печаль
ты просил весь город приезжали гниль
с вышел я в этом нос.
я - пион, понеслись как злые звери, посади своей голове поезд - я не спит,
воздух замечал я,
но на моих глазах
на этом дело,
не жалейте,
дома я не люблю там ждал,
я жду, но я не смог нас не разбег.
```

Really, looks like mid-ranged author`s creative, just without any understanding of text.
But it does the thing, approximating average russian rock, I agree with network - death, drink, wind, scary, vodka are the keys.
