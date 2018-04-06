import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

	public static void main(String[] args) {
		List<List<String>> trainingData = new ArrayList<List<String>>();

		trainingData.add(Arrays.asList("Coughing", "Male", "Adult", "Discharged"));
		trainingData.add(Arrays.asList("Coughing", "Female", "Teen", "Discharged"));
		trainingData.add(Arrays.asList("Headache", "Male", "Child", "Discharged"));
		trainingData.add(Arrays.asList("Headache", "Male", "Teen", "Discharged"));
		trainingData.add(Arrays.asList("Hiccups", "Female", "Adult", "Discharged"));
		trainingData.add(Arrays.asList("Sneezing", "Male", "Teen", "Discharged"));
		trainingData.add(Arrays.asList("Sneezing", "Female", "Child", "Admitted"));
		trainingData.add(Arrays.asList("Sneezing", "Male", "Child", "Admitted"));
		trainingData.add(Arrays.asList("Hiccups", "Female", "Teen", "Admitted"));
		trainingData.add(Arrays.asList("Coughing", "Female", "Adult", "Admitted"));

		AdaBoost predictor = new AdaBoost(trainingData, "Discharged");

		// System.out.println(predictor.Learners );

		predictor.Learners.stream().forEach(learner -> System.out.println(learner.ToString()));

		double vote = predictor.MakePrediction(new String[] { "Coughing", "Male", "Child" });
		System.out.println("The final vote is " + vote);
		System.out.println("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

		vote = predictor.MakePrediction(new String[] { "Headache", "Female", "Child" });
		System.out.println("The final vote is " + vote);
		System.out.println("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

		vote = predictor.MakePrediction(new String[] { "Headache", "", "Child" });
		System.out.println("The final vote is " + vote);
		System.out.println("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

		vote = predictor.MakePrediction(new String[] { "Coughing", "Female", "Adult" });
		System.out.println("The final vote is " + vote);
		System.out.println("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

	}
}
