import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class AdaBoost {
	/*
	 * rawTrain is the dataset with the last column having the outcome you're
	 * looking for. positiveOutcome is the value in the last column you want to
	 * predict for. You'll be given back a function that you can use to pass in
	 * "questions"
	 */

	public AdaBoost(List<List<String>> rawTrain, String positiveOutcome) {
		this.Learners = this.TrainData(rawTrain, positiveOutcome);
	}

	public List<Learner> Learners;

	public double MakePrediction(String[] testData) {
		System.out.println();
		// loop through each learner to see if it's needed.
		// if it is then accumulate the predicted * alpha

		return this.Learners.stream().filter(learner -> learner.Desc == testData[learner.Feature])
				.collect(Collectors.summingDouble(learner -> learner.Predicted * learner.Alpha));
	}

	public List<Learner> TrainData(List<List<String>> rawTrain, String positiveOutcome) {
		// Generate a values matrix
		List<List<String>> values = new ArrayList<List<String>>();

		int rows = rawTrain.size();

		int cols = rawTrain.get(0).size();

		for (int col = 0; col < cols; col++) {
			values.add(new ArrayList<String>());
			for (int row = 0; row < rows; row++) {
				if (!values.get(col).contains((rawTrain.get(row).get(col)))) {
					values.get(col).add(rawTrain.get(row).get(col));
				}
			}
		}

		// Make Learners
		int dataRows = rawTrain.size();
		int dataCols = rawTrain.get(0).size();
		List<Learner> learners = new ArrayList<Learner>();

		int valueCount = values.size();
		for (int featureIndex = 0; featureIndex < valueCount - 1; featureIndex++) {
			int featureCount = values.get(featureIndex).size();
			for (int valueIndex = 0; valueIndex < featureCount; valueIndex++) {
				String currentValue = values.get(featureIndex).get(valueIndex);

				// find how often each value occurs and what it's outcome was
				// return whether the value resulted more in a Discharged or Admitted
				// create a learner and add it to list or learners. Don't add if we couldn't
				// figure out a Discharged or Admitted.
				int plusOne = 0;
				int minusOne = 0;
				int relevant = 0;

				for (int dataRow = 0; dataRow < dataRows; dataRow++) {
					if (rawTrain.get(dataRow).get(featureIndex) == currentValue) {
						relevant++;
						if (rawTrain.get(dataRow).get(dataCols - 1) != positiveOutcome) {
							minusOne++;
						} else {
							plusOne++;
						}
					}
				}

				if (relevant != 0 && plusOne != minusOne) {
					// which one had more?
					int predicted = (plusOne > minusOne) ? 1 : -1;

					int epsilon;
					if (predicted == 1) {
						epsilon = minusOne * (1 / dataRows);
					} else {
						epsilon = plusOne * (1 / dataRows);
					}

					Learner newLearner = new Learner();
					newLearner.Desc = values.get(featureIndex).get(valueIndex);
					newLearner.Feature = featureIndex;
					newLearner.Predicted = predicted;
					newLearner.Epsilon = epsilon;

					learners.add(newLearner);
				}
			}
		}

		int learnerCount = learners.size();
		double[] trainWeights = new double[dataRows];
		int lastColumn = rawTrain.get(0).size() - 1;
		double startWeight = 1.0 / dataRows;
		// Initialize starting training weights
		for (int i = 0; i < dataRows; i++) {
			trainWeights[i] = startWeight;
		}

		// loop for however many learners there are.
		for (int learnIndex = 0; learnIndex < learnerCount; learnIndex++) {
			// update epsilons
			// loop through each learner and find it in the data
			// if the predicted value doesn't match then accumulate the training weight.
			// when all is said and done, assign the epsilon the accumulated training
			// weight.
			// lather, rinse, repeat for the next learner.
			for (int updateIndex = 0; updateIndex < learnerCount; updateIndex++) {
				Learner learner = learners.get(updateIndex);
				double ep = 0;
				for (int i = 0; i < dataRows; i++) {
					// if this row has the same value as the learner and the prediction is wrong
					// then accumulate the training weight.
					if (learner.Desc == rawTrain.get(i).get(learner.Feature)
							&& learner.Predicted != (rawTrain.get(i).get(lastColumn) == positiveOutcome ? 1 : -1)) {
						ep += trainWeights[i];
					}
				}

				learners.get(updateIndex).Epsilon = ep;
			}

			// find best learner. a.k.a. The unused one with the lowest epsilon.
			int bestLearner = -1;
			double lowestEpsilon = java.lang.Double.MAX_VALUE;
			for (int findIndex = 0; findIndex < learnerCount; findIndex++) {
				if (!learners.get(findIndex).Learned && learners.get(findIndex).Epsilon < lowestEpsilon) {
					lowestEpsilon = learners.get(findIndex).Epsilon;
					bestLearner = findIndex;
				}
			}

			// assign to something not zero. otherwise we get divide by zero later. Also,
			// the smaller this number is, the bigger 0 becomes. (there's probably a better
			// way to explain that)
			// ReSharper disable once CompareOfFloatsByEqualityOperator
			if (lowestEpsilon == 0) {
				lowestEpsilon = .000001; //// should maybe use double.Epsilon;
			}

			learners.get(bestLearner).Learned = true;
			double alpha = 0.5 * Math.log((1.0 - lowestEpsilon) / lowestEpsilon); // increases greatly the further
																					// epsilon was from .5
			learners.get(bestLearner).Alpha = alpha;

			// update training weights by finding training data that matches the learner and
			// scale it.
			Learner bLearner = learners.get(bestLearner);
			for (int i = 0; i < dataRows; i++) {
				if (bLearner.Desc == rawTrain.get(i).get(bLearner.Feature)) {
					trainWeights[i] = trainWeights[i] * Math.exp(-alpha
							* (rawTrain.get(i).get(lastColumn) == positiveOutcome ? 1 : -1) * bLearner.Predicted);
				}
			}

			// total the training weights then divide each weight by total.
			double weightTotals = 0;
			for (int i = 0; i < dataRows; i++) {
				weightTotals += trainWeights[i];
			}

			for (int i = 0; i < dataRows; i++) {
				trainWeights[i] = trainWeights[i] / weightTotals;
			}

			// Do that all over again for the next best learner until they've all been used.
		}

		return learners;
	}
}
