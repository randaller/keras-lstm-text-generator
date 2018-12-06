<?php require_once("functions.php"); ?>
<!doctype html>
<html>
<head>
        <title>Here AI generates text</title>
        <script src="jquery-3.3.1.min.js" crossorigin="anonymous"></script>

        <link rel="stylesheet" href="css/bootstrap.min.css" crossorigin="anonymous">
        <script src="js/bootstrap.min.js" crossorigin="anonymous"></script>

        <!-- script src="https://unpkg.com/keras-js"></script -->
        <script src="keras.min.js"></script>
        <script src="discrete.js"></script>

        <style>
                body {
                        padding-top: 24px;
                }

                .starter-template {
                        padding: 10px 15px;
                        text-align: center;
                }

                #result {
                        font-size: 16px;
                        border: 1px solid #808080;
                        border-radius: 5px;
                }
        </style>
</head>

<body>

<script>
        model = new KerasJS.Model({
                filepath: "<?php echo $model_file; ?>",
                gpu      : true
        });

        function change_model(sel)
        {
                window.location.href = "?model=" + sel.value;
        }

        <?php echo $chars; ?>

        const MAXLEN = <?php echo $seq_length; ?>;
        const CHARCOUNT = chars.length;

        sample = [];
        for (var i = 0; i < MAXLEN; i++)
        {
                digit = Math.round(Math.random() * chars.length);				
                for (var j = 0; j < CHARCOUNT; j++)
                {
                        if (j == digit)
                        {
                                sample.push(1.0);
                        }
                        else
                        {
                                sample.push(0.0);
                        }
                }
        }

        // console.log(sample);

        function sleep(ms)
        {
                return new Promise(function (resolve)
                {
                        setTimeout(resolve, ms)
                });
        }

        async function predict_100()
        {
                gen_chars = $("#count_selector").val();
                gen_chars = parseInt(gen_chars);

                for (var j = 0; j < gen_chars; j++)
                {
                        predict();
                        await sleep(1);
                }
        }

        function sampler(preds, temperature)
        {
                a = new Array(preds.length);
                for (var i = 0; i < preds.length; i++)
                {
                        a[i] = preds[i];
                }
                preds = a;

                // preds = nj.log(preds) / temperature;
                for (var i = 0; i < preds.length; i++)
                {
                        preds[i] = Math.log(preds[i]);
                }

                for (var i = 0; i < preds.length; i++)
                {
                        preds[i] = preds[i] / temperature;
                }

                // exp_preds = nj.exp(preds);
                exp_preds = new Array(preds.length);
                for (var i = 0; i < preds.length; i++)
                {
                        exp_preds[i] = Math.exp(preds[i]);
                }

                // preds = exp_preds / nj.sum(exp_preds);
                sum = 0;
                for (var i = 0; i < exp_preds.length; i++)
                {
                        sum += exp_preds[i];
                }

                for (var i = 0; i < exp_preds.length; i++)
                {
                        preds[i] = exp_preds[i] / sum;
                }

                // probas = nj.random.multinomial(1, preds, 1);
                var probas = SJS.Multinomial(1, preds);
                probas.draw();
                z = probas.sample(1);

                probas = z[0];

                // return nj.argmax(probas)
                max = -1;
                digit = null;
                for (var i = 0; i < probas.length; i++)
                {
                        if (probas[i] > max)
                        {
                                max = probas[i];
                                digit = i
                        }
                }

                return digit;
        }

        function predict()
        {
                // console.log("Sample: ", sample);

                model.ready()
                        .then(function ()
                        {
                                return model.predict({
                                        'input' : new Float32Array(sample)
                                })
                        })
                        .then(function (outputData)
                        {
                                const predictions = outputData.output;

                                // console.log(predictions);

                                temp = $("#temp_selector").val();
                                temp = parseFloat(temp);

                                digit = sampler(predictions, temp);

                                $("#result").append(chars[digit]);
                                <?php if ( $model_words ) echo '$("#result").append(" ");'; ?>

                                if (digit != null)
                                {
                                        sample.splice(0, CHARCOUNT);

                                        for (var j = 0; j < CHARCOUNT; j++)
                                        {
                                                if (j == digit)
                                                {
                                                        sample.push(1.0);
                                                }
                                                else
                                                {
                                                        sample.push(0.0);
                                                }
                                        }
                                }
                        })
                        .catch(function (error)
                        {
                                console.log(error);
                        });
        }

		function assign_seed()
		{
			seed_val = $("#seed").val().toLowerCase();
			
			to_shift = MAXLEN - seed_val.length;
			for (var i = 0; i < to_shift; i++)
			{
				sample.splice(0, CHARCOUNT);
				for (var j = 0; j < CHARCOUNT; j++)
				{
					sample.push(0.0);
				}
			}
			
			for (var i = 0; i < seed_val.length; i++) {
				next_char = seed_val.charAt(i);
				digit = chars.indexOf( next_char );
				sample.splice(0, CHARCOUNT);
				for (var j = 0; j < CHARCOUNT; j++)
				{
						if (j == digit)
						{
								sample.push(1.0);
						}
						else
						{
								sample.push(0.0);
						}
				}
			}
			
			$("#result").append( '<i>' + seed_val + '</i>' );
			
			// console.log(sample);
		}
		
        model.ready().then(function() { $("#result").text(""); } );
</script>

<div class="container">
        <div class="starter-template">
                <img src="logo.jpg"/><br/><br/>
                <form class="form-inline">

                        <label for="selector">Model:&nbsp;&nbsp;</label>
                        <select id="selector" class="form-control" onchange="change_model(this);">
                                <?php echo $selector; ?>
                        </select>

                        &nbsp;&nbsp;&nbsp;

                        <label for="temp_selector">Temperature:&nbsp;&nbsp;</label>
                        <select id="temp_selector" class="form-control">
								<option value="0.20">0.20 (most original)</option>
								<option value="0.25">0.25</option>
								<option value="0.30">0.30</option>
								<option value="0.35">0.35</option>
								<option value="0.40">0.40</option>
								<option value="0.45">0.45</option>
                                <option value="0.50">0.50 (more original)</option>
                                <option value="0.55">0.55</option>
                                <option value="0.60">0.60</option>
                                <option value="0.65">0.65</option>
                                <option value="0.70">0.70</option>
                                <option value="0.75">0.75</option>
                                <option value="0.80" selected>0.80 (neutral)</option>
                                <option value="0.85">0.85</option>
                                <option value="0.90">0.90</option>
                                <option value="0.95">0.95</option>
                                <option value="1.00">1.00</option>
                                <option value="1.05">1.05</option>
                                <option value="1.10">1.10 (more fantasy)</option>
								<option value="1.15">1.15</option>
                        </select>

                        &nbsp;&nbsp;&nbsp;

                        <label for="count_selector">Count:&nbsp;&nbsp;</label>
                        <select id="count_selector" class="form-control">
                                <option value="100" selected>100</option>
                                <option value="200">200</option>
                                <option value="500">500</option>
                                <option value="1000">1000</option>
                                <option value="2000">2000</option>
                                <option value="5000">5000</option>
                        </select>

                        &nbsp;&nbsp;&nbsp;

                        <input type="button" value="generate predictions" class="btn btn-primary" onclick="predict_100();"/>

                </form>

                <br/>

                <div id="result">Loading model, please wait...</div>

                <br/>

                <input type="button" value="generate predictions" class="btn btn-lg btn-primary" onclick="predict_100();"/>
                &nbsp;&nbsp;
                <input type="button" value="clear output" class="btn btn-lg btn-info" onclick="$('#result').empty();"/>
				&nbsp;&nbsp;
				<input type="button" value="reload model" class="btn btn-lg btn-warning" onclick="location.reload();"/>

                <!-- br/><br/>Tip: hit F5 to reload model and automatically regenerate seed -->
				
				<?php if (!$model_words): ?>
				<br/><br/>
				<input type="text" maxlength="<?php echo $seq_length; ?>" id="seed" style="width:480px;">
				<br/><br/>
				<input type="button" value="assign seed" class="btn btn-lg btn-primary" onclick="assign_seed();"/>
				<?php endif; ?>
        </div>
</div>

</body>
</html>
