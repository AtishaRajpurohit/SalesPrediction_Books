# DrFirst assessment

To proceed with answering the question of whether the company will be able to pay the loan back and be able to meet the new purchase order, the equation of value needs to be stated.

This equation consists of the total costs and total revenue:
### Total Costs :
    t_org: Total value of original books purchased+
    t_new: Total value of new books planning to be purchased+
    0.6$ * num_t_last: number of total books sent for last assortment+
    0.6$ * num_np_last: number of books NOT PURCHASED for last assortment+
    0.6$ * num_t_next: number of total books sent for next assortment +
    0.6$ * num_np_next: number of books NOT PURCHASED for next assortment (Assessed using an ML model)+
    
### Total Revenue : 
    t_p_last: Total value of the number of books PURCHASED for last assortment+
    t_p_next: Total value of the number of books PURCHASED for next assortment (Assessed using an ML model)
    
### Model

RandomForest - Since the data is qualitative and mainly categorical , it makes sense to use decision trees, with
an ensemble method and hence random forests. The below performs a Random Forest on the model for a number if parameters.
To aid with the code being run quicker this has already been run, and the best parameters have been chosen for the model.
The model has been tested using cross validation.

Finally proceeding to computing the equation of value. Any equation of value is incomplete without the rate of interest.
### There are 2 rates of interest here:
    Interest on the loan
    Interest on the revenue generated from the books
    
### Assumptions for the equation of value: 
1. The loan was purchased one month ago at month 0, is payable in 6 months as a lumpsum.
2. The last month assortment was sent for month 0.
3. The last month assortment will be sent a month from the present, hence month 2.
4. The next purchase will be assumed to be made on month 6, when the loan is due.

Interest on shipping costs have been ignored for materiality.    
It must be noted that other costs, such as inventory costs, cost of books being damaged when being sent back , etc have been ignored.
